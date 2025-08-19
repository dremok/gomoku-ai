"""
Tests for Board class.
"""
import numpy as np
import pytest
from gomoku.core.board import Board


def test_board_initialization():
    """Test that a board is initialized correctly."""
    board = Board()
    
    # Check board size
    assert board.size == 15
    
    # Check board state shape
    assert board.state.shape == (15, 15)
    
    # Check that all cells are empty (0)
    assert np.all(board.state == 0)
    
    # Check data type
    assert board.state.dtype == np.int8


def test_valid_move_placement():
    """Test that valid moves are placed correctly."""
    board = Board()
    
    # Test black stone placement
    result = board.apply_move(7, 7, 1)  # Center position, black
    assert result == True
    assert board.state[7, 7] == 1
    
    # Test white stone placement  
    result = board.apply_move(0, 0, -1)  # Corner position, white
    assert result == True
    assert board.state[0, 0] == -1
    
    # Test edge positions
    result = board.apply_move(14, 14, 1)  # Bottom-right corner
    assert result == True
    assert board.state[14, 14] == 1
    
    # Verify other positions remain empty
    assert board.state[1, 1] == 0
    assert board.state[5, 5] == 0


def test_invalid_move_rejection():
    """Test that invalid moves are rejected correctly."""
    board = Board()
    
    # Test out of bounds moves
    assert board.apply_move(-1, 5, 1) == False  # Negative row
    assert board.apply_move(5, -1, 1) == False  # Negative col
    assert board.apply_move(15, 5, 1) == False  # Row too large
    assert board.apply_move(5, 15, 1) == False  # Col too large
    assert board.apply_move(20, 20, 1) == False  # Both out of bounds
    
    # Test invalid player values
    assert board.apply_move(5, 5, 0) == False   # Invalid player
    assert board.apply_move(5, 5, 2) == False   # Invalid player
    assert board.apply_move(5, 5, -2) == False  # Invalid player
    
    # Test occupied cell
    board.apply_move(7, 7, 1)  # Place a stone
    assert board.apply_move(7, 7, -1) == False  # Try to overwrite with white
    assert board.apply_move(7, 7, 1) == False   # Try to overwrite with black
    
    # Verify board state unchanged after invalid moves
    assert board.state[7, 7] == 1  # Original stone still there
    assert np.sum(board.state != 0) == 1  # Only one stone on board


def test_empty_board_legal_moves():
    """Test that empty board returns all 225 positions as legal moves."""
    board = Board()
    
    legal_moves = board.get_legal_moves()
    
    # Check count
    assert len(legal_moves) == 225  # 15 * 15
    
    # Check all positions are included
    expected_positions = [(r, c) for r in range(15) for c in range(15)]
    assert set(legal_moves) == set(expected_positions)
    
    # Verify positions are valid tuples
    for move in legal_moves:
        assert isinstance(move, tuple)
        assert len(move) == 2
        row, col = move
        assert 0 <= row < 15
        assert 0 <= col < 15


def test_partial_board_legal_moves():
    """Test legal moves on board with some stones placed."""
    board = Board()
    
    # Place some stones
    board.apply_move(7, 7, 1)   # Center
    board.apply_move(0, 0, -1)  # Corner
    board.apply_move(14, 14, 1) # Opposite corner
    
    legal_moves = board.get_legal_moves()
    
    # Should have 225 - 3 = 222 legal moves
    assert len(legal_moves) == 222
    
    # Occupied positions should not be in legal moves
    occupied_positions = [(7, 7), (0, 0), (14, 14)]
    for pos in occupied_positions:
        assert pos not in legal_moves
    
    # Some empty positions should still be legal
    empty_positions = [(7, 6), (1, 1), (13, 13)]
    for pos in empty_positions:
        assert pos in legal_moves
    
    # Verify all returned positions are actually empty
    for row, col in legal_moves:
        assert board.state[row, col] == 0


def test_full_board_legal_moves():
    """Test legal moves on completely full board."""
    board = Board()
    
    # Fill entire board alternating black and white
    player = 1  # Start with black
    for row in range(15):
        for col in range(15):
            board.apply_move(row, col, player)
            player *= -1  # Alternate between 1 and -1
    
    legal_moves = board.get_legal_moves()
    
    # Should have no legal moves
    assert len(legal_moves) == 0
    assert legal_moves == []
    
    # Verify board is actually full
    assert np.all(board.state != 0)
    assert np.sum(board.state != 0) == 225