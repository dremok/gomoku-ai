"""
Tests for DuelingDQN model.
"""
import pytest
import torch
import numpy as np
from gomoku.ai.models.dqn_model import DuelingDQN
from gomoku.ai.models.state_encoding import encode_game_state, get_legal_moves_mask
from gomoku.core.game import Game


def test_dqn_model_initialization():
    """Test DQN model initializes with correct parameters."""
    model = DuelingDQN()
    
    # Check default parameters
    assert model.board_size == 15
    assert model.input_channels == 4
    assert model.conv_filters == 64
    assert model.num_actions == 225
    
    # Check that all layers are created
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'conv2') 
    assert hasattr(model, 'conv3')
    assert hasattr(model, 'conv4')
    assert hasattr(model, 'value_head')
    assert hasattr(model, 'advantage_head')


def test_dqn_model_custom_parameters():
    """Test DQN model with custom parameters."""
    model = DuelingDQN(board_size=9, input_channels=3, conv_filters=32)
    
    assert model.board_size == 9
    assert model.input_channels == 3
    assert model.conv_filters == 32
    assert model.num_actions == 81  # 9x9


def test_dqn_forward_pass_shape():
    """Test that forward pass produces correct output shape."""
    model = DuelingDQN()
    batch_size = 4
    
    # Create random input of correct shape
    x = torch.randn(batch_size, 4, 15, 15)
    
    # Forward pass
    q_values = model(x)
    
    # Check output shape
    assert q_values.shape == (batch_size, 225)
    assert q_values.dtype == torch.float32


def test_dqn_forward_pass_single_sample():
    """Test forward pass with single sample (batch size 1)."""
    model = DuelingDQN()
    
    # Single sample
    x = torch.randn(1, 4, 15, 15)
    q_values = model(x)
    
    assert q_values.shape == (1, 225)


def test_dqn_forward_pass_with_mask():
    """Test forward pass with legal moves mask."""
    model = DuelingDQN()
    batch_size = 2
    
    x = torch.randn(batch_size, 4, 15, 15)
    
    # Create a legal moves mask (some moves legal, some not)
    legal_mask = torch.zeros(batch_size, 225, dtype=torch.bool)
    legal_mask[0, :100] = True  # First sample: positions 0-99 legal
    legal_mask[1, 50:150] = True  # Second sample: positions 50-149 legal
    
    q_values = model(x, legal_moves_mask=legal_mask)
    
    assert q_values.shape == (batch_size, 225)
    
    # Check that illegal moves have very negative values
    illegal_penalty = -1e6
    assert torch.all(q_values[0, 100:] <= illegal_penalty + 1)  # Illegal moves for sample 0
    assert torch.all(q_values[1, :50] <= illegal_penalty + 1)   # Illegal moves for sample 1
    assert torch.all(q_values[1, 150:] <= illegal_penalty + 1)  # Illegal moves for sample 1


def test_dqn_dueling_architecture():
    """Test that the dueling architecture is working correctly."""
    model = DuelingDQN()
    
    # Create two identical states
    x1 = torch.randn(1, 4, 15, 15)
    x2 = x1.clone()
    
    q_values1 = model(x1)
    q_values2 = model(x2)
    
    # Identical inputs should produce identical outputs
    assert torch.allclose(q_values1, q_values2)


def test_dqn_with_real_game_state():
    """Test DQN with real game state encoding."""
    model = DuelingDQN()
    
    # Create a real game state
    game = Game()
    game.make_move(7, 7)  # Black
    game.make_move(7, 8)  # White
    
    # Encode state
    state = encode_game_state(game, last_move=(7, 8))
    state_batch = state.unsqueeze(0)  # Add batch dimension
    
    # Get legal moves mask
    legal_mask = get_legal_moves_mask(game).unsqueeze(0)
    
    # Forward pass
    q_values = model(state_batch, legal_moves_mask=legal_mask)
    
    assert q_values.shape == (1, 225)
    
    # Verify that occupied positions have very negative Q-values
    from gomoku.ai.models.state_encoding import encode_action_coordinates
    occupied_77 = encode_action_coordinates(7, 7)
    occupied_78 = encode_action_coordinates(7, 8)
    
    illegal_penalty = -1e6
    assert q_values[0, occupied_77] <= illegal_penalty + 1
    assert q_values[0, occupied_78] <= illegal_penalty + 1


def test_dqn_gradient_flow():
    """Test that gradients flow correctly through the network."""
    model = DuelingDQN()
    
    # Create input and target
    x = torch.randn(2, 4, 15, 15, requires_grad=True)
    target = torch.randn(2, 225)
    
    # Forward pass
    q_values = model(x)
    
    # Compute loss
    loss = F.mse_loss(q_values, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_dqn_batch_normalization():
    """Test that batch normalization works correctly."""
    model = DuelingDQN()
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 4, 15, 15)
        q_values = model(x)
        assert q_values.shape == (batch_size, 225)


def test_dqn_eval_mode():
    """Test that model works correctly in evaluation mode."""
    model = DuelingDQN()
    
    # Run some forward passes first to stabilize batch norm running averages
    dummy_x = torch.randn(4, 4, 15, 15)
    model.train()
    for _ in range(10):
        _ = model(dummy_x)
    
    x = torch.randn(1, 4, 15, 15)
    
    # Now test consistency in eval mode
    model.eval()
    with torch.no_grad():
        q_eval1 = model(x)
        q_eval2 = model(x)
    
    # Multiple eval passes should be identical
    assert q_eval1.shape == q_eval2.shape
    assert torch.allclose(q_eval1, q_eval2)
    
    # Should work with different batch sizes in eval mode
    x_batch = torch.randn(3, 4, 15, 15)
    q_batch = model(x_batch)
    assert q_batch.shape == (3, 225)


def test_dqn_value_advantage_combination():
    """Test that value and advantage are combined correctly."""
    model = DuelingDQN()
    
    # Create input
    x = torch.randn(2, 4, 15, 15)
    
    # Get intermediate outputs by modifying the forward pass temporarily
    with torch.no_grad():
        batch_size = x.size(0)
        
        # Feature extraction
        features = x
        features = F.relu(model.bn1(model.conv1(features)))
        features = F.relu(model.bn2(model.conv2(features)))
        features = F.relu(model.bn3(model.conv3(features)))
        features = F.relu(model.bn4(model.conv4(features)))
        features = features.view(batch_size, -1)
        
        # Get value and advantage
        value = model.value_head(features)
        advantage = model.advantage_head(features)
        
        # Manual combination (same as in forward pass)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        expected_q = value + advantage - advantage_mean
    
    # Get actual Q-values
    actual_q = model(x)
    
    # Should be very close (allowing for small numerical differences)
    assert torch.allclose(actual_q, expected_q, rtol=1e-5)


def test_dqn_legal_mask_mean_calculation():
    """Test that legal moves mask affects advantage mean calculation correctly."""
    model = DuelingDQN()
    
    x = torch.randn(1, 4, 15, 15)
    
    # Create mask with only a few legal moves
    legal_mask = torch.zeros(1, 225, dtype=torch.bool)
    legal_mask[0, [0, 1, 2, 10, 20]] = True  # Only 5 legal moves
    
    q_masked = model(x, legal_moves_mask=legal_mask)
    q_unmasked = model(x)
    
    # Results should be different due to different mean calculations
    assert not torch.allclose(q_masked, q_unmasked)
    
    # Legal moves should have reasonable values, illegal moves should be penalized
    legal_positions = [0, 1, 2, 10, 20]
    illegal_penalty = -1e6
    
    for pos in legal_positions:
        assert q_masked[0, pos] > illegal_penalty + 100  # Legal moves not penalized
    
    # Check some illegal positions
    illegal_positions = [3, 4, 5, 100, 200]
    for pos in illegal_positions:
        assert q_masked[0, pos] <= illegal_penalty + 1  # Illegal moves penalized


# Import F for the gradient test
import torch.nn.functional as F