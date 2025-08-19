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


def test_horizontal_5_in_a_row_win():
    """Test detection of horizontal 5-in-a-row wins."""
    board = Board()
    
    # Test black horizontal win in middle row
    for col in range(5, 10):  # Positions (7, 5) through (7, 9)
        board.apply_move(7, col, 1)
    
    # Check win detection for each position in the line
    for col in range(5, 10):
        winner = board.check_winner(7, col)
        assert winner == 1, f"Should detect black win at position (7, {col})"
    
    # Test white horizontal win in different row
    board2 = Board()
    for col in range(0, 5):  # Positions (3, 0) through (3, 4) 
        board2.apply_move(3, col, -1)
    
    # Check win detection for white
    for col in range(0, 5):
        winner = board2.check_winner(3, col)
        assert winner == -1, f"Should detect white win at position (3, {col})"


def test_horizontal_4_in_a_row_no_win():
    """Test that 4-in-a-row does not trigger a win."""
    board = Board()
    
    # Place only 4 black stones horizontally
    for col in range(5, 9):  # Positions (7, 5) through (7, 8)
        board.apply_move(7, col, 1)
    
    # Should not detect a win
    for col in range(5, 9):
        winner = board.check_winner(7, col)
        assert winner is None, f"Should not detect win with only 4 stones at (7, {col})"


def test_horizontal_overline_no_win():
    """Test that horizontal overlines (6+ stones) do NOT count as wins."""
    board = Board()
    
    # Place 6 black stones horizontally (overline)
    for col in range(4, 10):  # Positions (7, 4) through (7, 9) - 6 stones
        board.apply_move(7, col, 1)
    
    # Should not detect a win for any position (overline rule)
    for col in range(4, 10):
        winner = board.check_winner(7, col)
        assert winner is None, f"Should not detect win with overline at position (7, {col})"
    
    # Test 7 stones (even longer overline)
    board2 = Board()
    for col in range(3, 10):  # Positions (5, 3) through (5, 9) - 7 stones
        board2.apply_move(5, col, -1)
    
    # Should not detect a win for white either
    for col in range(3, 10):
        winner = board2.check_winner(5, col)
        assert winner is None, f"Should not detect white win with longer overline at (5, {col})"


def test_horizontal_embedded_5_in_overline():
    """Test that a 5-in-a-row embedded within a longer line is NOT a win."""
    board = Board()
    
    # Create pattern: empty, 6 stones, empty (positions 1-6 have stones)
    # This tests that even though there are 5 consecutive stones within the 6,
    # it should not be detected as a win due to overline
    for col in range(1, 7):  # 6 stones at (7, 1) through (7, 6)
        board.apply_move(7, col, 1)
    
    # No position should register as a win
    for col in range(1, 7):
        winner = board.check_winner(7, col)
        assert winner is None, f"Embedded 5 in overline should not win at (7, {col})"


def test_horizontal_edge_cases():
    """Test horizontal win detection at board boundaries."""
    board = Board()
    
    # Test win at left edge of board
    for col in range(0, 5):  # Positions (7, 0) through (7, 4)
        board.apply_move(7, col, 1)
    
    winner = board.check_winner(7, 2)  # Check middle of the line
    assert winner == 1, "Should detect win at left edge of board"
    
    # Test win at right edge of board  
    board2 = Board()
    for col in range(10, 15):  # Positions (7, 10) through (7, 14)
        board2.apply_move(7, col, -1)
    
    winner = board2.check_winner(7, 12)  # Check middle of the line
    assert winner == -1, "Should detect win at right edge of board"


def test_horizontal_no_false_positives():
    """Test that non-winning patterns don't trigger false wins."""
    board = Board()
    
    # Test scattered stones (not contiguous)
    positions = [(7, 1), (7, 3), (7, 5), (7, 7), (7, 9)]
    for row, col in positions:
        board.apply_move(row, col, 1)
        
    # Should not detect wins
    for row, col in positions:
        winner = board.check_winner(row, col)
        assert winner is None, f"Should not detect win for scattered stones at ({row}, {col})"
    
    # Test mixed players in line
    board2 = Board()
    board2.apply_move(5, 5, 1)   # Black
    board2.apply_move(5, 6, -1)  # White  
    board2.apply_move(5, 7, 1)   # Black
    board2.apply_move(5, 8, 1)   # Black
    board2.apply_move(5, 9, 1)   # Black
    
    # Should not detect wins for any position
    for col in range(5, 10):
        winner = board2.check_winner(5, col)
        assert winner is None, f"Should not detect win in mixed line at (5, {col})"


def test_vertical_5_in_a_row_win():
    """Test detection of vertical 5-in-a-row wins."""
    board = Board()
    
    # Test black vertical win in middle column
    for row in range(5, 10):  # Positions (5, 7) through (9, 7)
        board.apply_move(row, 7, 1)
    
    # Check win detection for each position in the line
    for row in range(5, 10):
        winner = board.check_winner(row, 7)
        assert winner == 1, f"Should detect black vertical win at position ({row}, 7)"
    
    # Test white vertical win in different column
    board2 = Board()
    for row in range(0, 5):  # Positions (0, 3) through (4, 3)
        board2.apply_move(row, 3, -1)
    
    # Check win detection for white
    for row in range(0, 5):
        winner = board2.check_winner(row, 3)
        assert winner == -1, f"Should detect white vertical win at position ({row}, 3)"


def test_vertical_overline_no_win():
    """Test that vertical overlines (6+ stones) do NOT count as wins."""
    board = Board()
    
    # Place 6 black stones vertically (overline)
    for row in range(4, 10):  # Positions (4, 7) through (9, 7) - 6 stones
        board.apply_move(row, 7, 1)
    
    # Should not detect a win for any position (overline rule)
    for row in range(4, 10):
        winner = board.check_winner(row, 7)
        assert winner is None, f"Should not detect vertical win with overline at ({row}, 7)"


def test_diagonal_5_in_a_row_win():
    """Test detection of diagonal (↘) 5-in-a-row wins."""
    board = Board()
    
    # Test black diagonal win (↘ direction)
    positions = [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    for row, col in positions:
        board.apply_move(row, col, 1)
    
    # Check win detection for each position in the line
    for row, col in positions:
        winner = board.check_winner(row, col)
        assert winner == 1, f"Should detect black diagonal win at position ({row}, {col})"
    
    # Test white diagonal win in different location
    board2 = Board()
    positions2 = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    for row, col in positions2:
        board2.apply_move(row, col, -1)
    
    # Check win detection for white
    for row, col in positions2:
        winner = board2.check_winner(row, col)
        assert winner == -1, f"Should detect white diagonal win at position ({row}, {col})"


def test_diagonal_overline_no_win():
    """Test that diagonal overlines (6+ stones) do NOT count as wins."""
    board = Board()
    
    # Place 6 black stones diagonally (overline)
    positions = [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    for row, col in positions:
        board.apply_move(row, col, 1)
    
    # Should not detect a win for any position (overline rule)
    for row, col in positions:
        winner = board.check_winner(row, col)
        assert winner is None, f"Should not detect diagonal win with overline at ({row}, {col})"


def test_anti_diagonal_5_in_a_row_win():
    """Test detection of anti-diagonal (↙) 5-in-a-row wins."""
    board = Board()
    
    # Test black anti-diagonal win (↙ direction)
    positions = [(5, 9), (6, 8), (7, 7), (8, 6), (9, 5)]
    for row, col in positions:
        board.apply_move(row, col, 1)
    
    # Check win detection for each position in the line
    for row, col in positions:
        winner = board.check_winner(row, col)
        assert winner == 1, f"Should detect black anti-diagonal win at position ({row}, {col})"
    
    # Test white anti-diagonal win in different location
    board2 = Board()
    positions2 = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    for row, col in positions2:
        board2.apply_move(row, col, -1)
    
    # Check win detection for white
    for row, col in positions2:
        winner = board2.check_winner(row, col)
        assert winner == -1, f"Should detect white anti-diagonal win at position ({row}, {col})"


def test_anti_diagonal_overline_no_win():
    """Test that anti-diagonal overlines (6+ stones) do NOT count as wins."""
    board = Board()
    
    # Place 6 black stones anti-diagonally (overline)
    positions = [(4, 10), (5, 9), (6, 8), (7, 7), (8, 6), (9, 5)]
    for row, col in positions:
        board.apply_move(row, col, 1)
    
    # Should not detect a win for any position (overline rule)
    for row, col in positions:
        winner = board.check_winner(row, col)
        assert winner is None, f"Should not detect anti-diagonal win with overline at ({row}, {col})"


def test_all_directions_edge_cases():
    """Test win detection at board boundaries for all directions."""
    
    # Test vertical win at top edge
    board1 = Board()
    for row in range(0, 5):  # Top edge vertical
        board1.apply_move(row, 7, 1)
    winner = board1.check_winner(2, 7)
    assert winner == 1, "Should detect vertical win at top edge"
    
    # Test vertical win at bottom edge
    board2 = Board()
    for row in range(10, 15):  # Bottom edge vertical
        board2.apply_move(row, 7, -1)
    winner = board2.check_winner(12, 7)
    assert winner == -1, "Should detect vertical win at bottom edge"
    
    # Test diagonal win at top-left corner
    board3 = Board()
    positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    for row, col in positions:
        board3.apply_move(row, col, 1)
    winner = board3.check_winner(2, 2)
    assert winner == 1, "Should detect diagonal win at top-left"
    
    # Test anti-diagonal win at top-right corner
    board4 = Board()
    positions = [(0, 14), (1, 13), (2, 12), (3, 11), (4, 10)]
    for row, col in positions:
        board4.apply_move(row, col, -1)
    winner = board4.check_winner(2, 12)
    assert winner == -1, "Should detect anti-diagonal win at top-right"


def test_mixed_directions_no_false_wins():
    """Test that patterns across different directions don't create false wins."""
    board = Board()
    
    # Create a cross pattern (+ shape) - should not win
    center_row, center_col = 7, 7
    
    # Horizontal line (3 stones)
    board.apply_move(center_row, center_col - 1, 1)
    board.apply_move(center_row, center_col, 1)
    board.apply_move(center_row, center_col + 1, 1)
    
    # Vertical line (3 stones, center overlaps)
    board.apply_move(center_row - 1, center_col, 1)
    board.apply_move(center_row + 1, center_col, 1)
    
    # Should not detect any wins (no single direction has 5)
    test_positions = [
        (center_row, center_col - 1),
        (center_row, center_col),
        (center_row, center_col + 1),
        (center_row - 1, center_col),
        (center_row + 1, center_col),
    ]
    
    for row, col in test_positions:
        winner = board.check_winner(row, col)
        assert winner is None, f"Cross pattern should not win at ({row}, {col})"


def test_full_board_no_winner_is_draw():
    """Test that a full board with no winner is detected as a draw."""
    board = Board()
    
    # Fill the board with a pattern that prevents any 5-in-a-rows
    # Use alternating pattern that changes every 3 positions to break up lines
    for row in range(15):
        for col in range(15):
            # Pattern that ensures no 5 consecutive stones of same color in any direction
            if (row + col + row // 3 + col // 3) % 2 == 0:
                current_player = 1  # Black
            else:
                current_player = -1  # White
            board.apply_move(row, col, current_player)
    
    # Verify board is full
    assert len(board.get_legal_moves()) == 0, "Board should be full"
    
    # Should be detected as a draw
    assert board.is_draw() == True, "Full board with no winner should be a draw"


def test_board_with_legal_moves_not_draw():
    """Test that boards with remaining legal moves are not draws."""
    # Test empty board
    board1 = Board()
    assert board1.is_draw() == False, "Empty board should not be a draw"
    assert len(board1.get_legal_moves()) == 225, "Empty board should have 225 legal moves"
    
    # Test partially filled board
    board2 = Board()
    positions = [(7, 7), (7, 8), (8, 7), (8, 8)]  # Place 4 stones
    for i, (row, col) in enumerate(positions):
        player = 1 if i % 2 == 0 else -1
        board2.apply_move(row, col, player)
    
    assert board2.is_draw() == False, "Partially filled board should not be a draw"
    assert len(board2.get_legal_moves()) == 221, "Should have 221 legal moves remaining"
    
    # Test nearly full board (1 move remaining)
    board3 = Board()
    player = 1
    for row in range(15):
        for col in range(15):
            if row == 7 and col == 7:  # Leave one position empty
                continue
            # Use alternating pattern to prevent wins
            if (row * 15 + col) % 4 < 2:
                current_player = 1
            else:
                current_player = -1
            board3.apply_move(row, col, current_player)
    
    assert board3.is_draw() == False, "Nearly full board should not be a draw"
    assert len(board3.get_legal_moves()) == 1, "Should have 1 legal move remaining"


def test_board_with_winner_not_draw():
    """Test that boards with a winner are not draws, even if nearly/completely full."""
    # Test simple board with clear winner and remaining moves
    board1 = Board()
    # Create horizontal win for black
    for col in range(5, 10):  # 5 stones: (7,5) to (7,9)
        board1.apply_move(7, col, 1)
    
    # Add some other non-interfering moves
    board1.apply_move(5, 5, -1)
    board1.apply_move(6, 6, -1)
    
    assert board1.is_draw() == False, "Board with winner should not be a draw"
    winner = board1.check_winner(7, 7)  # Check middle of winning line
    assert winner == 1, "Should detect black as winner"
    
    # Test board with winner but no remaining legal moves would be unusual,
    # but let's test a simpler case - a board with a winner and some moves
    board2 = Board()
    
    # Create a clear vertical win for white
    for row in range(2, 7):  # 5 stones: (2,10) to (6,10)
        board2.apply_move(row, 10, -1)
        
    # Add some other stones that don't interfere  
    board2.apply_move(0, 0, 1)
    board2.apply_move(1, 1, 1)
    board2.apply_move(14, 14, 1)
    
    assert board2.is_draw() == False, "Board with winner should not be a draw"
    winner = board2.check_winner(4, 10)  # Check middle of winning line
    assert winner == -1, "Should detect white as winner"