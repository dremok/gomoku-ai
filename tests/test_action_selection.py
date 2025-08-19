"""
Tests for action selection utilities.
"""
import pytest
import torch
import random
import numpy as np
from gomoku.ai.models.action_selection import (
    epsilon_greedy_action,
    epsilon_greedy_action_batch,
    greedy_action,
    random_legal_action,
    softmax_action,
    action_to_coordinates,
    coordinates_to_action
)


def test_epsilon_greedy_action_fully_greedy():
    """Test ε-greedy with ε=0 (fully greedy)."""
    # Create Q-values where position 10 has highest value among legal moves
    q_values = torch.randn(225)
    q_values[10] = 10.0  # Highest value
    q_values[5] = 8.0    # Second highest
    
    # Legal moves mask - only positions 5, 10, 20 are legal
    legal_mask = torch.zeros(225, dtype=torch.bool)
    legal_mask[[5, 10, 20]] = True
    
    # With ε=0, should always select position 10 (highest Q-value among legal moves)
    rng = random.Random(42)
    for _ in range(10):
        action = epsilon_greedy_action(q_values, legal_mask, epsilon=0.0, rng=rng)
        assert action == 10


def test_epsilon_greedy_action_fully_random():
    """Test ε-greedy with ε=1 (fully random)."""
    q_values = torch.randn(225)
    q_values[10] = 10.0  # Highest value (should be ignored with ε=1)
    
    # Legal moves mask
    legal_moves = [5, 10, 15, 20, 25]
    legal_mask = torch.zeros(225, dtype=torch.bool)
    legal_mask[legal_moves] = True
    
    # With ε=1, should select randomly from legal moves
    rng = random.Random(42)
    actions = []
    for _ in range(50):
        action = epsilon_greedy_action(q_values, legal_mask, epsilon=1.0, rng=rng)
        actions.append(action)
        assert action in legal_moves  # Should always be legal
    
    # Should have some variety (not always the same action)
    unique_actions = set(actions)
    assert len(unique_actions) > 1, "Should select different actions with ε=1"


def test_epsilon_greedy_action_mixed():
    """Test ε-greedy with mixed exploration (ε=0.3)."""
    q_values = torch.randn(225)
    q_values[10] = 10.0  # Clear best choice
    
    legal_moves = [5, 10, 15, 20]
    legal_mask = torch.zeros(225, dtype=torch.bool)
    legal_mask[legal_moves] = True
    
    rng = random.Random(42)
    actions = []
    greedy_count = 0
    
    for _ in range(100):
        action = epsilon_greedy_action(q_values, legal_mask, epsilon=0.3, rng=rng)
        actions.append(action)
        assert action in legal_moves  # Should always be legal
        if action == 10:  # Best action
            greedy_count += 1
    
    # With ε=0.3, should exploit ~70% of the time
    # Allow some variance due to randomness
    assert 60 <= greedy_count <= 80, f"Expected ~70 greedy actions, got {greedy_count}"


def test_epsilon_greedy_action_no_legal_moves():
    """Test ε-greedy raises error with no legal moves."""
    q_values = torch.randn(225)
    legal_mask = torch.zeros(225, dtype=torch.bool)  # No legal moves
    
    with pytest.raises(ValueError, match="No legal moves available"):
        epsilon_greedy_action(q_values, legal_mask, epsilon=0.5)


def test_epsilon_greedy_action_single_legal_move():
    """Test ε-greedy with only one legal move."""
    q_values = torch.randn(225)
    legal_mask = torch.zeros(225, dtype=torch.bool)
    legal_mask[42] = True  # Only position 42 is legal
    
    rng = random.Random(42)
    
    # Should always select the only legal move regardless of ε
    for epsilon in [0.0, 0.5, 1.0]:
        action = epsilon_greedy_action(q_values, legal_mask, epsilon=epsilon, rng=rng)
        assert action == 42


def test_epsilon_greedy_action_batch():
    """Test batch ε-greedy action selection."""
    batch_size = 3
    num_actions = 225
    
    # Create batch of Q-values
    q_values = torch.randn(batch_size, num_actions)
    q_values[0, 10] = 10.0  # Sample 0: best action is 10
    q_values[1, 50] = 10.0  # Sample 1: best action is 50
    q_values[2, 100] = 10.0  # Sample 2: best action is 100
    
    # Create legal moves masks
    legal_masks = torch.zeros(batch_size, num_actions, dtype=torch.bool)
    legal_masks[0, [5, 10, 15]] = True   # Sample 0: legal moves 5, 10, 15
    legal_masks[1, [40, 50, 60]] = True  # Sample 1: legal moves 40, 50, 60
    legal_masks[2, [90, 100, 110]] = True  # Sample 2: legal moves 90, 100, 110
    
    rng = random.Random(42)
    
    # Test fully greedy (ε=0)
    actions = epsilon_greedy_action_batch(q_values, legal_masks, epsilon=0.0, rng=rng)
    assert actions.shape == (batch_size,)
    assert actions[0] == 10   # Best legal action for sample 0
    assert actions[1] == 50   # Best legal action for sample 1
    assert actions[2] == 100  # Best legal action for sample 2


def test_epsilon_greedy_action_batch_no_legal_moves():
    """Test batch ε-greedy raises error when sample has no legal moves."""
    batch_size = 2
    q_values = torch.randn(batch_size, 225)
    
    legal_masks = torch.zeros(batch_size, 225, dtype=torch.bool)
    legal_masks[0, [10, 20]] = True  # Sample 0 has legal moves
    # Sample 1 has no legal moves
    
    with pytest.raises(ValueError, match="No legal moves available for sample 1"):
        epsilon_greedy_action_batch(q_values, legal_masks, epsilon=0.5)


def test_greedy_action():
    """Test fully greedy action selection."""
    q_values = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    legal_mask = torch.tensor([True, True, False, True, True])  # Position 2 illegal
    
    # Should select position 1 (highest Q-value among legal moves)
    action = greedy_action(q_values, legal_mask)
    assert action == 1


def test_random_legal_action():
    """Test random legal action selection."""
    legal_moves = [5, 10, 15, 20, 25]
    legal_mask = torch.zeros(225, dtype=torch.bool)
    legal_mask[legal_moves] = True
    
    rng = random.Random(42)
    actions = []
    
    for _ in range(50):
        action = random_legal_action(legal_mask, rng=rng)
        actions.append(action)
        assert action in legal_moves
    
    # Should have variety
    unique_actions = set(actions)
    assert len(unique_actions) > 1


def test_random_legal_action_no_legal_moves():
    """Test random action raises error with no legal moves."""
    legal_mask = torch.zeros(225, dtype=torch.bool)
    
    with pytest.raises(ValueError, match="No legal moves available"):
        random_legal_action(legal_mask)


def test_softmax_action():
    """Test softmax action selection."""
    # Create Q-values where position 1 is clearly best
    q_values = torch.tensor([1.0, 10.0, 2.0, 3.0, 4.0])
    legal_mask = torch.tensor([True, True, True, True, True])
    
    rng = random.Random(42)
    actions = []
    
    # With low temperature, should mostly select position 1
    for _ in range(100):
        action = softmax_action(q_values, legal_mask, temperature=0.1, rng=rng)
        actions.append(action)
        assert 0 <= action < 5
    
    # Position 1 should be selected most often
    assert actions.count(1) > 50  # Should be heavily biased toward best action


def test_softmax_action_high_temperature():
    """Test softmax with high temperature (more random)."""
    q_values = torch.tensor([1.0, 10.0, 2.0, 3.0, 4.0])
    legal_mask = torch.tensor([True, True, True, True, True])
    
    rng = random.Random(42)
    actions = []
    
    # With high temperature, should be more uniform
    for _ in range(200):
        action = softmax_action(q_values, legal_mask, temperature=10.0, rng=rng)
        actions.append(action)
    
    # Should have reasonable variety
    unique_actions = set(actions)
    assert len(unique_actions) >= 3  # Should explore different actions


def test_softmax_action_invalid_temperature():
    """Test softmax raises error with invalid temperature."""
    q_values = torch.randn(5)
    legal_mask = torch.ones(5, dtype=torch.bool)
    
    with pytest.raises(ValueError, match="Temperature must be positive"):
        softmax_action(q_values, legal_mask, temperature=0.0)
    
    with pytest.raises(ValueError, match="Temperature must be positive"):
        softmax_action(q_values, legal_mask, temperature=-1.0)


def test_action_coordinate_conversion():
    """Test conversion between action indices and coordinates."""
    board_size = 15
    
    # Test specific positions
    test_cases = [
        (0, (0, 0)),      # Top-left
        (14, (0, 14)),    # Top-right
        (210, (14, 0)),   # Bottom-left
        (224, (14, 14)),  # Bottom-right
        (112, (7, 7)),    # Center
        (23, (1, 8)),     # Random position
    ]
    
    for action_idx, (expected_row, expected_col) in test_cases:
        # Test action_to_coordinates
        row, col = action_to_coordinates(action_idx, board_size)
        assert row == expected_row
        assert col == expected_col
        
        # Test coordinates_to_action (round trip)
        recovered_action = coordinates_to_action(row, col, board_size)
        assert recovered_action == action_idx


def test_coordinate_conversion_round_trip():
    """Test that coordinate conversions are perfect inverses."""
    board_size = 15
    
    # Test all positions
    for row in range(board_size):
        for col in range(board_size):
            # Convert coordinates to action and back
            action_idx = coordinates_to_action(row, col, board_size)
            recovered_row, recovered_col = action_to_coordinates(action_idx, board_size)
            
            assert recovered_row == row
            assert recovered_col == col
    
    # Test all action indices
    for action_idx in range(board_size * board_size):
        # Convert action to coordinates and back
        row, col = action_to_coordinates(action_idx, board_size)
        recovered_action = coordinates_to_action(row, col, board_size)
        
        assert recovered_action == action_idx


def test_action_selection_reproducibility():
    """Test that action selection is reproducible with same random seed."""
    q_values = torch.randn(225)
    legal_mask = torch.ones(225, dtype=torch.bool)
    legal_mask[:100] = False  # Only last 125 positions are legal
    
    # Same seed should produce same results
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    
    actions1 = []
    actions2 = []
    
    for _ in range(20):
        action1 = epsilon_greedy_action(q_values, legal_mask, epsilon=0.5, rng=rng1)
        action2 = epsilon_greedy_action(q_values, legal_mask, epsilon=0.5, rng=rng2)
        actions1.append(action1)
        actions2.append(action2)
    
    assert actions1 == actions2, "Same seed should produce identical action sequences"