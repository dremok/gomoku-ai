"""
Tests for DQN trainer.
"""
import pytest
import torch
import tempfile
import os
from gomoku.ai.models.dqn_model import DuelingDQN
from gomoku.ai.training.dqn_trainer import DQNTrainer
from gomoku.ai.training.replay_buffer import ReplayBuffer


def create_dummy_batch(batch_size=4, board_size=15):
    """Create dummy training batch for testing."""
    states = torch.randn(batch_size, 4, board_size, board_size)
    actions = torch.randint(0, board_size * board_size, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, board_size, board_size)
    dones = torch.randint(0, 2, (batch_size,)).bool()
    legal_masks = torch.ones(batch_size, board_size * board_size, dtype=torch.bool)
    next_legal_masks = torch.ones(batch_size, board_size * board_size, dtype=torch.bool)
    
    # Make some moves illegal for realism
    legal_masks[:, :50] = False  # First 50 positions illegal
    next_legal_masks[:, :25] = False  # First 25 positions illegal in next states
    
    return states, actions, rewards, next_states, dones, legal_masks, next_legal_masks


def test_dqn_trainer_initialization():
    """Test DQN trainer initializes correctly."""
    online_net = DuelingDQN(board_size=15)
    
    trainer = DQNTrainer(
        online_network=online_net,
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.01,
        batch_size=32,
        device='cpu'
    )
    
    assert trainer.online_network is not None
    assert trainer.target_network is not None
    assert trainer.gamma == 0.99
    assert trainer.tau == 0.01
    assert trainer.batch_size == 32
    assert trainer.training_steps == 0
    assert trainer.double_dqn == True


def test_dqn_trainer_with_custom_target_network():
    """Test DQN trainer with provided target network."""
    online_net = DuelingDQN(board_size=15)
    target_net = DuelingDQN(board_size=15)
    
    trainer = DQNTrainer(
        online_network=online_net,
        target_network=target_net,
        device='cpu'
    )
    
    assert trainer.online_network is online_net
    assert trainer.target_network is target_net


def test_dqn_trainer_compute_td_targets():
    """Test TD target computation."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, device='cpu')
    
    batch_size = 4
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 15, 15)
    next_legal_masks = torch.ones(batch_size, 225, dtype=torch.bool)
    dones = torch.tensor([False, True, False, False])
    
    td_targets = trainer.compute_td_targets(rewards, next_states, next_legal_masks, dones)
    
    assert td_targets.shape == (batch_size,)
    assert td_targets.dtype == torch.float32
    
    # For done states, target should just be the reward
    assert torch.allclose(td_targets[1], rewards[1], atol=1e-6)
    
    # For non-done states, target should include future value (reward + gamma * next_q)
    # Note: With untrained networks, Q-values can be negative, so we just check
    # that the computation is happening (target != immediate reward for non-terminal)
    assert td_targets[0] != rewards[0]  # Should include future value term
    assert td_targets[2] != rewards[2]  # Should not equal immediate reward alone


def test_dqn_trainer_double_dqn_vs_standard():
    """Test difference between Double DQN and standard DQN."""
    online_net = DuelingDQN(board_size=15)
    
    # Double DQN trainer
    trainer_double = DQNTrainer(online_network=online_net, double_dqn=True, device='cpu')
    
    # Standard DQN trainer  
    trainer_standard = DQNTrainer(online_network=online_net, double_dqn=False, device='cpu')
    
    batch_size = 4
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 15, 15)
    next_legal_masks = torch.ones(batch_size, 225, dtype=torch.bool)
    dones = torch.tensor([False, False, False, False])
    
    # Compute targets with both methods
    td_targets_double = trainer_double.compute_td_targets(rewards, next_states, next_legal_masks, dones)
    td_targets_standard = trainer_standard.compute_td_targets(rewards, next_states, next_legal_masks, dones)
    
    assert td_targets_double.shape == td_targets_standard.shape
    # They might be different due to different action selection methods
    # This test mainly ensures both methods work without error


def test_dqn_trainer_train_step():
    """Test single training step."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, device='cpu')
    
    states, actions, rewards, next_states, dones, legal_masks, next_legal_masks = create_dummy_batch()
    
    # Perform training step
    metrics = trainer.train_step(states, actions, rewards, next_states, dones, 
                               legal_masks, next_legal_masks)
    
    # Check metrics
    assert 'loss' in metrics
    assert 'q_mean' in metrics
    assert 'q_std' in metrics
    assert 'td_targets_mean' in metrics
    assert 'td_targets_std' in metrics
    
    assert isinstance(metrics['loss'], float)
    assert metrics['loss'] >= 0  # Loss should be non-negative
    
    # Check training step was incremented
    assert trainer.training_steps == 1


def test_dqn_trainer_target_network_update():
    """Test target network updates."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, tau=0.1, device='cpu')
    
    # Get initial target network parameters
    initial_target_params = [p.clone() for p in trainer.target_network.parameters()]
    
    # Modify online network parameters (simulate training)
    with torch.no_grad():
        for p in trainer.online_network.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    
    # Soft update target network
    trainer.update_target_network()
    
    # Check that target network parameters changed
    for initial_param, current_param in zip(initial_target_params, trainer.target_network.parameters()):
        assert not torch.allclose(initial_param, current_param)
    
    # Test hard update
    trainer.hard_update_target_network()
    
    # After hard update, target should match online exactly
    for online_param, target_param in zip(trainer.online_network.parameters(), 
                                        trainer.target_network.parameters()):
        assert torch.allclose(online_param, target_param)


def test_dqn_trainer_with_replay_buffer():
    """Test training with replay buffer."""
    online_net = DuelingDQN(board_size=15)
    replay_buffer = ReplayBuffer(capacity=1000, batch_size=4)
    
    trainer = DQNTrainer(
        online_network=online_net,
        replay_buffer=replay_buffer,
        batch_size=4,
        device='cpu'
    )
    
    # Add some dummy experiences to buffer
    for i in range(10):
        state = torch.randn(4, 15, 15)
        next_state = torch.randn(4, 15, 15) if i < 9 else None
        done = i == 9
        
        replay_buffer.add(
            state=state,
            action=i % 225,
            reward=float(i % 3 - 1),  # -1, 0, or 1
            next_state=next_state,
            done=done
        )
    
    # Should be able to train on batch
    assert trainer.replay_buffer.can_sample(trainer.batch_size)
    
    metrics = trainer.train_on_batch()
    assert metrics is not None
    assert 'loss' in metrics
    assert trainer.training_steps == 1


def test_dqn_trainer_insufficient_buffer_samples():
    """Test behavior when replay buffer has insufficient samples."""
    online_net = DuelingDQN(board_size=15)
    replay_buffer = ReplayBuffer(capacity=1000, batch_size=32)
    
    trainer = DQNTrainer(
        online_network=online_net,
        replay_buffer=replay_buffer,
        batch_size=32,
        device='cpu'
    )
    
    # Add only a few experiences (less than batch size)
    for i in range(5):
        state = torch.randn(4, 15, 15)
        replay_buffer.add(
            state=state,
            action=i,
            reward=0.0,
            next_state=state,
            done=False
        )
    
    # Should return None when not enough samples
    metrics = trainer.train_on_batch()
    assert metrics is None
    assert trainer.training_steps == 0


def test_dqn_trainer_training_stats():
    """Test training statistics tracking."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, device='cpu')
    
    initial_stats = trainer.get_training_stats()
    assert initial_stats['training_steps'] == 0
    assert initial_stats['avg_loss'] == 0.0
    
    # Perform a few training steps
    for _ in range(3):
        states, actions, rewards, next_states, dones, legal_masks, next_legal_masks = create_dummy_batch()
        trainer.train_step(states, actions, rewards, next_states, dones, 
                         legal_masks, next_legal_masks)
    
    final_stats = trainer.get_training_stats()
    assert final_stats['training_steps'] == 3
    assert final_stats['avg_loss'] > 0
    assert 'q_mean' in final_stats
    assert 'learning_rate' in final_stats


def test_dqn_trainer_gradient_clipping():
    """Test gradient clipping functionality."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, grad_clip=1.0, device='cpu')
    
    states, actions, rewards, next_states, dones, legal_masks, next_legal_masks = create_dummy_batch()
    
    # Training step should work with gradient clipping
    metrics = trainer.train_step(states, actions, rewards, next_states, dones, 
                               legal_masks, next_legal_masks)
    assert metrics is not None
    
    # Test with no gradient clipping
    trainer.grad_clip = 0.0
    metrics = trainer.train_step(states, actions, rewards, next_states, dones, 
                               legal_masks, next_legal_masks)
    assert metrics is not None


def test_dqn_trainer_learning_rate_adjustment():
    """Test learning rate adjustment."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, learning_rate=1e-3, device='cpu')
    
    # Check initial learning rate
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    assert initial_lr == 1e-3
    
    # Change learning rate
    trainer.set_learning_rate(5e-4)
    new_lr = trainer.optimizer.param_groups[0]['lr']
    assert new_lr == 5e-4


def test_dqn_trainer_save_load_checkpoint():
    """Test saving and loading training checkpoints."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, device='cpu')
    
    # Perform some training to create state
    states, actions, rewards, next_states, dones, legal_masks, next_legal_masks = create_dummy_batch()
    trainer.train_step(states, actions, rewards, next_states, dones, 
                     legal_masks, next_legal_masks)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
        
        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path, metadata={'test': 'value'})
        assert os.path.exists(checkpoint_path)
        
        # Create new trainer and load checkpoint
        new_net = DuelingDQN(board_size=15)
        new_trainer = DQNTrainer(online_network=new_net, device='cpu')
        
        metadata = new_trainer.load_checkpoint(checkpoint_path)
        
        # Check that state was restored
        assert new_trainer.training_steps == trainer.training_steps
        assert new_trainer.total_loss == trainer.total_loss
        assert metadata['test'] == 'value'
        
        # Check that network weights match
        for orig_param, loaded_param in zip(trainer.online_network.parameters(),
                                          new_trainer.online_network.parameters()):
            assert torch.allclose(orig_param, loaded_param)


def test_dqn_trainer_no_replay_buffer_error():
    """Test error when training without replay buffer."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, device='cpu')
    
    # Should raise error when trying to train without buffer
    with pytest.raises(ValueError, match="No replay buffer provided"):
        trainer.train_on_batch()


def test_dqn_trainer_device_handling():
    """Test that trainer handles device correctly."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, device='cpu')
    
    assert str(trainer.device) == 'cpu'
    assert next(trainer.online_network.parameters()).device.type == 'cpu'
    assert next(trainer.target_network.parameters()).device.type == 'cpu'
    
    # Test training with CPU tensors
    states, actions, rewards, next_states, dones, legal_masks, next_legal_masks = create_dummy_batch()
    metrics = trainer.train_step(states, actions, rewards, next_states, dones, 
                               legal_masks, next_legal_masks)
    assert metrics is not None


def test_dqn_trainer_terminal_state_handling():
    """Test proper handling of terminal states."""
    online_net = DuelingDQN(board_size=15)
    trainer = DQNTrainer(online_network=online_net, device='cpu')
    
    batch_size = 4
    states = torch.randn(batch_size, 4, 15, 15)
    actions = torch.randint(0, 225, (batch_size,))
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 15, 15)
    dones = torch.tensor([False, True, True, False])  # Mix of terminal and non-terminal
    legal_masks = torch.ones(batch_size, 225, dtype=torch.bool)
    next_legal_masks = torch.ones(batch_size, 225, dtype=torch.bool)
    
    # For terminal states, should handle gracefully
    metrics = trainer.train_step(states, actions, rewards, next_states, dones, 
                               legal_masks, next_legal_masks)
    assert metrics is not None
    
    # TD targets for terminal states should be close to immediate rewards
    td_targets = trainer.compute_td_targets(rewards, next_states, next_legal_masks, dones)
    
    # For terminal states (indices 1 and 2), targets should equal rewards
    assert torch.allclose(td_targets[1], rewards[1], atol=1e-6)
    assert torch.allclose(td_targets[2], rewards[2], atol=1e-6)