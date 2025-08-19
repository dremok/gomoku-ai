"""
Tests for replay buffer functionality.
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path
from gomoku.ai.training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Transition


def create_dummy_state(channels=4, size=15):
    """Create a dummy state tensor for testing."""
    return torch.randn(channels, size, size)


def test_replay_buffer_initialization():
    """Test replay buffer initializes correctly."""
    buffer = ReplayBuffer(capacity=1000, batch_size=32, seed=42)
    
    assert buffer.capacity == 1000
    assert buffer.batch_size == 32
    assert len(buffer) == 0
    assert not buffer.is_full()
    assert not buffer.can_sample()


def test_replay_buffer_add_single_transition():
    """Test adding single transitions to buffer."""
    buffer = ReplayBuffer(capacity=100, seed=42)
    
    state = create_dummy_state()
    next_state = create_dummy_state()
    
    buffer.add(state, action=42, reward=1.0, next_state=next_state, done=False)
    
    assert len(buffer) == 1
    assert buffer.total_added == 1
    assert not buffer.is_full()


def test_replay_buffer_add_terminal_transition():
    """Test adding terminal transition (no next state)."""
    buffer = ReplayBuffer(capacity=100, seed=42)
    
    state = create_dummy_state()
    
    buffer.add(state, action=42, reward=1.0, next_state=None, done=True)
    
    assert len(buffer) == 1
    assert buffer.total_added == 1


def test_replay_buffer_add_trajectory():
    """Test adding multiple transitions as trajectory."""
    buffer = ReplayBuffer(capacity=100, seed=42)
    
    trajectory = []
    for i in range(5):
        state = create_dummy_state()
        next_state = create_dummy_state() if i < 4 else None
        done = i == 4
        
        transition = Transition(
            state=state,
            action=i,
            reward=float(i),
            next_state=next_state,
            done=done
        )
        trajectory.append(transition)
    
    buffer.add_trajectory(trajectory)
    
    assert len(buffer) == 5
    assert buffer.total_added == 5


def test_replay_buffer_capacity_overflow():
    """Test buffer respects capacity limit."""
    buffer = ReplayBuffer(capacity=3, seed=42)
    
    # Add more transitions than capacity
    for i in range(5):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    assert len(buffer) == 3  # Should not exceed capacity
    assert buffer.total_added == 5  # But should track total added
    assert buffer.is_full()


def test_replay_buffer_sampling():
    """Test sampling from buffer."""
    buffer = ReplayBuffer(capacity=100, batch_size=4, seed=42)
    
    # Add some transitions
    for i in range(10):
        state = create_dummy_state()
        next_state = create_dummy_state()
        buffer.add(state, action=i, reward=float(i), next_state=next_state, done=False)
    
    assert buffer.can_sample()
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample()
    
    assert states.shape == (4, 4, 15, 15)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 4, 15, 15)
    assert dones.shape == (4,)
    
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.bool


def test_replay_buffer_sampling_with_terminal_states():
    """Test sampling when some transitions are terminal."""
    buffer = ReplayBuffer(capacity=100, batch_size=3, seed=42)
    
    # Add mix of terminal and non-terminal transitions
    for i in range(5):
        state = create_dummy_state()
        
        if i == 2:  # Terminal transition
            buffer.add(state, action=i, reward=1.0, next_state=None, done=True)
        else:
            next_state = create_dummy_state()
            buffer.add(state, action=i, reward=0.0, next_state=next_state, done=False)
    
    # Should handle terminal states gracefully
    states, actions, rewards, next_states, dones = buffer.sample()
    
    assert states.shape == (3, 4, 15, 15)
    assert next_states.shape == (3, 4, 15, 15)  # Zero tensors for terminal states


def test_replay_buffer_insufficient_samples():
    """Test error when trying to sample without enough transitions."""
    buffer = ReplayBuffer(capacity=100, batch_size=10, seed=42)
    
    # Add only 5 transitions
    for i in range(5):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    assert not buffer.can_sample()
    
    with pytest.raises(ValueError, match="Not enough samples"):
        buffer.sample()


def test_replay_buffer_custom_batch_size():
    """Test sampling with custom batch size."""
    buffer = ReplayBuffer(capacity=100, batch_size=4, seed=42)
    
    # Add enough transitions
    for i in range(20):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    # Sample with different batch size
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=8)
    
    assert states.shape[0] == 8
    assert actions.shape[0] == 8


def test_replay_buffer_statistics():
    """Test buffer statistics."""
    buffer = ReplayBuffer(capacity=5, seed=42)
    
    # Add some transitions
    for i in range(3):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    stats = buffer.get_stats()
    
    assert stats['size'] == 3
    assert stats['capacity'] == 5
    assert stats['total_added'] == 3
    assert stats['utilization'] == 0.6
    assert not stats['is_full']


def test_replay_buffer_clear():
    """Test clearing buffer."""
    buffer = ReplayBuffer(capacity=100, seed=42)
    
    # Add some transitions
    for i in range(5):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    assert len(buffer) == 5
    
    buffer.clear()
    
    assert len(buffer) == 0
    assert buffer.total_added == 0


def test_replay_buffer_save_load():
    """Test saving and loading buffer."""
    buffer = ReplayBuffer(capacity=100, batch_size=16, seed=42)
    
    # Add some transitions
    original_transitions = []
    for i in range(10):
        state = create_dummy_state()
        next_state = create_dummy_state()
        buffer.add(state, action=i, reward=float(i), next_state=next_state, done=False)
        original_transitions.append((state, i, float(i), next_state, False))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save buffer
        save_path = os.path.join(tmpdir, 'test_buffer.pkl')
        buffer.save(save_path)
        
        # Check files exist
        assert os.path.exists(save_path + '.gz')
        
        # Load into new buffer
        new_buffer = ReplayBuffer(capacity=50)  # Different capacity
        new_buffer.load(save_path + '.gz')
        
        # Check loaded values
        assert len(new_buffer) == 10
        assert new_buffer.capacity == 100  # Should use saved capacity
        assert new_buffer.batch_size == 16
        assert new_buffer.total_added == 10


def test_replay_buffer_load_from_file():
    """Test loading buffer using class method."""
    buffer = ReplayBuffer(capacity=100, batch_size=8, seed=42)
    
    # Add some transitions
    for i in range(5):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_buffer.pkl')
        buffer.save(save_path)
        
        # Load using class method
        loaded_buffer = ReplayBuffer.load_from_file(save_path + '.gz')
        
        assert len(loaded_buffer) == 5
        assert loaded_buffer.capacity == 100
        assert loaded_buffer.batch_size == 8


def test_replay_buffer_load_nonexistent_file():
    """Test loading from nonexistent file raises error."""
    buffer = ReplayBuffer()
    
    with pytest.raises(FileNotFoundError):
        buffer.load('nonexistent_file.gz')


def test_prioritized_replay_buffer_initialization():
    """Test prioritized replay buffer initializes correctly."""
    buffer = PrioritizedReplayBuffer(
        capacity=1000, 
        batch_size=32, 
        alpha=0.6, 
        beta=0.4,
        seed=42
    )
    
    assert buffer.capacity == 1000
    assert buffer.batch_size == 32
    assert buffer.alpha == 0.6
    assert buffer.beta == 0.4
    assert len(buffer) == 0


def test_prioritized_replay_buffer_add_with_priority():
    """Test adding transitions with priorities."""
    buffer = PrioritizedReplayBuffer(capacity=100, seed=42)
    
    state = create_dummy_state()
    next_state = create_dummy_state()
    
    buffer.add(state, action=42, reward=1.0, next_state=next_state, done=False, priority=2.5)
    
    assert len(buffer) == 1
    assert len(buffer.priorities) == 1
    assert buffer.max_priority == 2.5


def test_prioritized_replay_buffer_update_priorities():
    """Test updating priorities after sampling."""
    buffer = PrioritizedReplayBuffer(capacity=100, batch_size=2, seed=42)
    
    # Add some transitions
    for i in range(5):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    # Sample and update priorities
    batch = buffer.sample()
    states, actions, rewards, next_states, dones, weights, indices = batch
    
    assert len(indices) == 2
    assert weights.shape == (2,)
    
    # Update priorities
    new_priorities = [1.5, 2.0]
    buffer.update_priorities(indices, new_priorities)
    
    # Check that priorities were updated (allow small epsilon for numerical stability)
    for idx, priority in zip(indices, new_priorities):
        assert abs(buffer.priorities[idx] - priority) < 1e-5


def test_prioritized_replay_buffer_sampling_shapes():
    """Test that prioritized sampling returns correct shapes."""
    buffer = PrioritizedReplayBuffer(capacity=100, batch_size=4, seed=42)
    
    # Add transitions
    for i in range(10):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    batch = buffer.sample()
    states, actions, rewards, next_states, dones, weights, indices = batch
    
    assert states.shape == (4, 4, 15, 15)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 4, 15, 15)
    assert dones.shape == (4,)
    assert weights.shape == (4,)
    assert len(indices) == 4


def test_prioritized_replay_buffer_beta_annealing():
    """Test that beta increases over time."""
    buffer = PrioritizedReplayBuffer(
        capacity=100, 
        batch_size=2, 
        beta=0.4,
        beta_increment=0.1,
        seed=42
    )
    
    # Add some transitions
    for i in range(5):
        state = create_dummy_state()
        buffer.add(state, action=i, reward=0.0, next_state=state, done=False)
    
    initial_beta = buffer.beta
    
    # Sample multiple times to trigger beta annealing
    for _ in range(3):
        buffer.sample()
    
    assert buffer.beta > initial_beta
    assert buffer.beta <= 1.0  # Should be clamped to 1.0