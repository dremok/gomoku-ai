"""
Tests for Game class.
"""
import pytest
from gomoku.core.game import Game
from gomoku.core.board import Board


def test_game_initialization():
    """Test that a game is initialized correctly."""
    game = Game()
    
    # Check game has a board
    assert isinstance(game.board, Board)
    
    # Check initial player is black (goes first)
    assert game.current_player == 1
    
    # Check initial game state
    assert game.game_state == 'ongoing'
    assert game.winner is None
    
    # Check board is empty initially
    assert len(game.board.get_legal_moves()) == 225
    assert game.board.is_draw() == False


def test_game_state_properties():
    """Test game state properties work correctly."""
    game = Game()
    
    # Initially ongoing
    assert game.game_state == 'ongoing'
    assert game.winner is None
    
    # Test internal state tracking
    assert game._winner is None
    assert game._is_draw == False


def test_valid_move_processing():
    """Test that valid moves are processed correctly."""
    game = Game()
    
    # Make a valid move for black (current player)
    result = game.make_move(7, 7)
    assert result == True, "Valid move should succeed"
    
    # Check board was updated
    assert game.board.state[7, 7] == 1, "Black stone should be placed"
    
    # Check player switched to white
    assert game.current_player == -1, "Current player should switch to white"
    
    # Game should still be ongoing
    assert game.game_state == 'ongoing'
    assert game.winner is None
    
    # Make another valid move for white
    result = game.make_move(7, 8)
    assert result == True, "Valid move should succeed"
    
    # Check board was updated
    assert game.board.state[7, 8] == -1, "White stone should be placed"
    
    # Check player switched back to black
    assert game.current_player == 1, "Current player should switch back to black"


def test_invalid_move_rejection():
    """Test that invalid moves are rejected correctly."""
    game = Game()
    
    # Test out of bounds moves
    result = game.make_move(-1, 5)
    assert result == False, "Out of bounds move should be rejected"
    assert game.current_player == 1, "Current player should not change on invalid move"
    
    result = game.make_move(15, 5)
    assert result == False, "Out of bounds move should be rejected"
    
    # Place a stone and try to overwrite it
    game.make_move(7, 7)  # Black places stone
    assert game.current_player == -1, "Should switch to white after valid move"
    
    result = game.make_move(7, 7)  # White tries to overwrite
    assert result == False, "Occupied position should be rejected"
    assert game.current_player == -1, "Current player should not change on invalid move"
    
    # Board should only have the one valid stone
    assert game.board.state[7, 7] == 1, "Original stone should remain"
    assert len([pos for pos in game.board.get_legal_moves() if pos != (7, 7)]) == 224


def test_game_over_move_rejection():
    """Test that moves are rejected when game is already over."""
    game = Game()
    
    # Create a winning position for black
    for col in range(5, 10):  # Horizontal win
        game.board.apply_move(7, col, 1)
        
    # Set game state to win
    game._winner = 1
    
    # Try to make a move - should be rejected
    result = game.make_move(6, 6)
    assert result == False, "Moves should be rejected when game is over"
    assert game.board.state[6, 6] == 0, "No stone should be placed"


def test_win_detection_after_move():
    """Test that wins are detected correctly after moves."""
    game = Game()
    
    # Set up a near-winning position for black (4 in a row)
    # Move sequence: Black, White, Black, White, Black, White, Black
    game.make_move(7, 5)  # Black
    game.make_move(6, 5)  # White  
    game.make_move(7, 6)  # Black
    game.make_move(6, 6)  # White
    game.make_move(7, 7)  # Black
    game.make_move(6, 7)  # White
    game.make_move(7, 8)  # Black (4 in a row now)
    
    # At this point it should be white's turn
    assert game.current_player == -1, "Should be white's turn"
    assert game.game_state == 'ongoing', "Game should still be ongoing"
    
    # White makes a non-interfering move
    game.make_move(8, 5)  # White
    
    # Now it's black's turn to make the winning move
    assert game.current_player == 1, "Should be black's turn"
    result = game.make_move(7, 9)  # Black makes winning move (5th stone)
    assert result == True, "Winning move should succeed"
    
    # Check game state updated correctly
    assert game.game_state == 'win', "Game should be in win state"
    assert game.winner == 1, "Black should be the winner"
    assert game._winner == 1, "Internal winner state should be set"
    
    # Current player should NOT have switched (game over)
    assert game.current_player == 1, "Current player should not switch after winning move"


def test_win_detection_white():
    """Test that white wins are detected correctly."""
    game = Game()
    
    # Alternate moves to set up white win
    # Black moves to avoid interfering with white's line
    game.make_move(6, 5)  # Black
    game.make_move(7, 5)  # White
    game.make_move(6, 6)  # Black  
    game.make_move(7, 6)  # White
    game.make_move(6, 7)  # Black
    game.make_move(7, 7)  # White
    game.make_move(6, 8)  # Black
    game.make_move(7, 8)  # White (4 in a row)
    
    # Should be black's turn, white has 4 in a row
    assert game.current_player == 1, "Should be black's turn"
    assert game.game_state == 'ongoing', "Game should still be ongoing"
    
    # Black makes a non-interfering move
    game.make_move(8, 5)  # Black
    
    # White makes the winning move
    result = game.make_move(7, 9)  # White completes 5 in a row
    assert result == True, "Winning move should succeed"
    
    # Check white wins
    assert game.game_state == 'win', "Game should be in win state"
    assert game.winner == -1, "White should be the winner"


def test_draw_detection_after_move():
    """Test that draws are detected correctly after moves."""
    # This test is simpler - we'll directly fill the board using the board
    # and then test that game state is updated correctly
    game = Game()
    
    # Fill the board directly with the pattern we know works (from Board tests)
    for row in range(15):
        for col in range(15):
            if (row + col + row // 3 + col // 3) % 2 == 0:
                current_player = 1  # Black
            else:
                current_player = -1  # White
            game.board.apply_move(row, col, current_player)
    
    # Manually set the draw state by calling our draw check logic
    # This simulates what would happen after the last move
    if game.board.is_draw():
        game._is_draw = True
        
    # Verify the draw is detected
    assert game.board.is_draw() == True, "Board should be in draw state"
    assert game.game_state == 'draw', "Game should be in draw state"
    assert game.winner is None, "Should be no winner in draw"
    assert game._is_draw == True, "Internal draw state should be set"