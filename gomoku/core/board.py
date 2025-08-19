"""
Board implementation for Gomoku game.
"""
import numpy as np


class Board:
    """
    Represents a 15x15 Gomoku board.
    
    Board state representation:
    - 0: empty cell
    - 1: black stone
    - -1: white stone
    """
    
    def __init__(self):
        """Initialize an empty 15x15 board."""
        self.size = 15
        self.state = np.zeros((self.size, self.size), dtype=np.int8)
    
    def apply_move(self, row, col, player):
        """
        Apply a move to the board.
        
        Args:
            row (int): Row position (0-14)
            col (int): Column position (0-14)
            player (int): Player (1 for black, -1 for white)
            
        Returns:
            bool: True if move was applied successfully, False if invalid
        """
        # Validate coordinates are in bounds
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
            
        # Validate cell is empty
        if self.state[row, col] != 0:
            return False
            
        # Validate player value
        if player not in (1, -1):
            return False
            
        # Apply the move
        self.state[row, col] = player
        return True
    
    def get_legal_moves(self):
        """
        Get all legal move positions on the board.
        
        Returns:
            list: List of (row, col) tuples representing empty positions
        """
        legal_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.state[row, col] == 0:  # Empty cell
                    legal_moves.append((row, col))
        return legal_moves