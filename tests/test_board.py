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