"""
Game implementation for Gomoku.
"""
from .board import Board


class Game:
    """
    Manages a Gomoku game session.
    
    Handles turn management, game state, and coordinates between 
    the board and players.
    """
    
    def __init__(self):
        """Initialize a new Gomoku game."""
        self.board = Board()
        self.current_player = 1  # Black goes first (1=black, -1=white)
        self._winner = None
        self._is_draw = False
        
    @property
    def game_state(self):
        """
        Get the current game state.
        
        Returns:
            str: One of 'ongoing', 'win', 'draw'
        """
        if self._winner is not None:
            return 'win'
        elif self._is_draw:
            return 'draw'
        else:
            return 'ongoing'
            
    @property
    def winner(self):
        """
        Get the winner of the game.
        
        Returns:
            int or None: Winner (1 for black, -1 for white) or None if no winner
        """
        return self._winner
    
    def make_move(self, row, col):
        """
        Make a move for the current player.
        
        Args:
            row (int): Row position (0-14)
            col (int): Column position (0-14)
            
        Returns:
            bool: True if move was successful, False if invalid or game over
        """
        # Can't make moves if game is already over
        if self.game_state != 'ongoing':
            return False
            
        # Try to apply the move to the board
        if not self.board.apply_move(row, col, self.current_player):
            return False  # Invalid move (out of bounds, occupied, etc.)
            
        # Move was successful, check for game ending conditions
        winner = self.board.check_winner(row, col)
        if winner is not None:
            self._winner = winner
        elif self.board.is_draw():
            self._is_draw = True
        else:
            # Game continues, switch to next player
            self.current_player *= -1  # Switch between 1 and -1
            
        return True