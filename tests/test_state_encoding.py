"""
Tests for DQN state encoding utilities.
"""
import pytest
import torch
import numpy as np
from gomoku.core.game import Game
from gomoku.ai.models.state_encoding import (
    encode_game_state, 
    decode_action_index, 
    encode_action_coordinates,
    get_legal_moves_mask
)


def test_encode_game_state_empty_board():
    """Test state encoding on empty board."""
    game = Game()
    
    # Encode initial state (black to play, no moves)
    state = encode_game_state(game)
    
    # Check shape
    assert state.shape == (4, 15, 15)
    assert state.dtype == torch.float32
    
    # Channel 0: Current player stones (black) - should be all zeros
    assert torch.all(state[0] == 0)
    
    # Channel 1: Opponent stones (white) - should be all zeros  
    assert torch.all(state[1] == 0)
    
    # Channel 2: Turn plane - should be all 1s (black to play)
    assert torch.all(state[2] == 1.0)
    
    # Channel 3: Last-move plane - should be all zeros (no last move)
    assert torch.all(state[3] == 0)


def test_encode_game_state_with_moves():
    """Test state encoding with some moves played."""
    game = Game()
    
    # Make some moves
    game.make_move(7, 7)  # Black at center
    game.make_move(7, 8)  # White next to center
    
    # Encode state (now black to play again)
    state = encode_game_state(game, last_move=(7, 8))
    
    # Check shape
    assert state.shape == (4, 15, 15)
    
    # Channel 0: Current player stones (black)
    assert state[0, 7, 7] == 1.0  # Black stone at (7,7)
    assert state[0, 7, 8] == 0.0  # No black stone at (7,8)
    assert torch.sum(state[0]) == 1.0  # Only one black stone
    
    # Channel 1: Opponent stones (white)
    assert state[1, 7, 7] == 0.0  # No white stone at (7,7)
    assert state[1, 7, 8] == 1.0  # White stone at (7,8)
    assert torch.sum(state[1]) == 1.0  # Only one white stone
    
    # Channel 2: Turn plane - should be all 1s (black to play)
    assert torch.all(state[2] == 1.0)
    
    # Channel 3: Last-move plane - should mark (7,8)
    assert state[3, 7, 8] == 1.0
    assert torch.sum(state[3]) == 1.0  # Only one position marked


def test_encode_game_state_white_to_play():
    """Test state encoding when white is to play."""
    game = Game()
    game.make_move(7, 7)  # Black move, now white to play
    
    state = encode_game_state(game, last_move=(7, 7))
    
    # Channel 0: Current player stones (white) - should be empty
    assert torch.all(state[0] == 0)
    
    # Channel 1: Opponent stones (black) - should have (7,7)
    assert state[1, 7, 7] == 1.0
    assert torch.sum(state[1]) == 1.0
    
    # Channel 2: Turn plane - should be all 0s (white to play)
    assert torch.all(state[2] == 0.0)
    
    # Channel 3: Last-move plane - should mark (7,7)
    assert state[3, 7, 7] == 1.0


def test_decode_action_index():
    """Test action index decoding."""
    # Test corners and center
    assert decode_action_index(0) == (0, 0)      # Top-left
    assert decode_action_index(14) == (0, 14)    # Top-right  
    assert decode_action_index(210) == (14, 0)   # Bottom-left
    assert decode_action_index(224) == (14, 14)  # Bottom-right
    assert decode_action_index(112) == (7, 7)    # Center
    
    # Test some random positions
    assert decode_action_index(23) == (1, 8)     # Row 1, col 8
    assert decode_action_index(195) == (13, 0)   # Row 13, col 0


def test_encode_action_coordinates():
    """Test action coordinate encoding."""
    # Test corners and center
    assert encode_action_coordinates(0, 0) == 0
    assert encode_action_coordinates(0, 14) == 14
    assert encode_action_coordinates(14, 0) == 210
    assert encode_action_coordinates(14, 14) == 224
    assert encode_action_coordinates(7, 7) == 112
    
    # Test some random positions
    assert encode_action_coordinates(1, 8) == 23
    assert encode_action_coordinates(13, 0) == 195


def test_action_encoding_round_trip():
    """Test that encoding and decoding actions are inverse operations."""
    board_size = 15
    
    # Test all valid positions
    for row in range(board_size):
        for col in range(board_size):
            # Encode then decode
            action_idx = encode_action_coordinates(row, col, board_size)
            decoded_row, decoded_col = decode_action_index(action_idx, board_size)
            
            assert decoded_row == row
            assert decoded_col == col
            
    # Test all valid action indices
    for action_idx in range(board_size * board_size):
        # Decode then encode
        row, col = decode_action_index(action_idx, board_size)
        encoded_idx = encode_action_coordinates(row, col, board_size)
        
        assert encoded_idx == action_idx


def test_get_legal_moves_mask_empty_board():
    """Test legal moves mask on empty board."""
    game = Game()
    mask = get_legal_moves_mask(game)
    
    # All positions should be legal on empty board
    assert mask.shape == (225,)
    assert mask.dtype == torch.bool
    assert torch.all(mask == True)  # All 225 positions legal


def test_get_legal_moves_mask_partial_board():
    """Test legal moves mask on partially filled board."""
    game = Game()
    
    # Make some moves
    game.make_move(7, 7)  # Black
    game.make_move(7, 8)  # White
    game.make_move(8, 7)  # Black
    
    mask = get_legal_moves_mask(game)
    
    # Should have 225 - 3 = 222 legal moves
    assert torch.sum(mask) == 222
    
    # Occupied positions should be False
    assert mask[encode_action_coordinates(7, 7)] == False
    assert mask[encode_action_coordinates(7, 8)] == False  
    assert mask[encode_action_coordinates(8, 7)] == False
    
    # Some unoccupied positions should be True
    assert mask[encode_action_coordinates(0, 0)] == True
    assert mask[encode_action_coordinates(14, 14)] == True
    assert mask[encode_action_coordinates(6, 6)] == True


def test_get_legal_moves_mask_full_board():
    """Test legal moves mask on completely filled board."""
    game = Game()
    
    # Fill entire board
    for row in range(15):
        for col in range(15):
            game.board.apply_move(row, col, 1 if (row + col) % 2 == 0 else -1)
    
    mask = get_legal_moves_mask(game)
    
    # No positions should be legal
    assert torch.sum(mask) == 0
    assert torch.all(mask == False)


def test_state_encoding_consistency():
    """Test that state encoding is consistent across multiple calls."""
    game = Game()
    game.make_move(7, 7)
    game.make_move(8, 8)
    
    # Encode same state multiple times
    state1 = encode_game_state(game, last_move=(8, 8))
    state2 = encode_game_state(game, last_move=(8, 8))
    
    # Should be identical
    assert torch.equal(state1, state2)


def test_state_encoding_last_move_optional():
    """Test state encoding works with and without last_move."""
    game = Game()
    game.make_move(7, 7)
    
    # Without last_move
    state_no_last = encode_game_state(game)
    assert torch.all(state_no_last[3] == 0)  # Channel 3 should be all zeros
    
    # With last_move
    state_with_last = encode_game_state(game, last_move=(7, 7))
    assert state_with_last[3, 7, 7] == 1.0  # Should mark the position
    
    # Other channels should be identical
    assert torch.equal(state_no_last[0], state_with_last[0])
    assert torch.equal(state_no_last[1], state_with_last[1])
    assert torch.equal(state_no_last[2], state_with_last[2])