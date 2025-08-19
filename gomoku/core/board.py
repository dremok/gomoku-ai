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
    
    def _check_line_win(self, row, col, player, dr, dc):
        """
        Check for a win in a specific direction from the given position.
        Implements Standard Gomoku (5-only) rule: exactly 5 stones, no overlines.
        
        Args:
            row (int): Starting row position
            col (int): Starting column position  
            player (int): Player to check (1 or -1)
            dr (int): Row direction (-1, 0, 1)
            dc (int): Column direction (-1, 0, 1)
            
        Returns:
            bool: True if exactly 5 contiguous stones with no overline
        """
        # Count stones in negative direction
        count_neg = 0
        r, c = row - dr, col - dc
        while (0 <= r < self.size and 0 <= c < self.size and 
               self.state[r, c] == player):
            count_neg += 1
            r, c = r - dr, c - dc
        
        # Count stones in positive direction  
        count_pos = 0
        r, c = row + dr, col + dc
        while (0 <= r < self.size and 0 <= c < self.size and
               self.state[r, c] == player):
            count_pos += 1
            r, c = r + dr, c + dc
            
        # Total count including the stone at (row, col)
        total_count = count_neg + 1 + count_pos
        
        # Must be exactly 5 for a win
        if total_count != 5:
            return False
            
        # Check for overline: verify no extension beyond the 5
        # Check negative end
        neg_end_r, neg_end_c = row - dr * count_neg - dr, col - dc * count_neg - dc
        if (0 <= neg_end_r < self.size and 0 <= neg_end_c < self.size and
            self.state[neg_end_r, neg_end_c] == player):
            return False  # Overline detected
            
        # Check positive end
        pos_end_r, pos_end_c = row + dr * count_pos + dr, col + dc * count_pos + dc
        if (0 <= pos_end_r < self.size and 0 <= pos_end_c < self.size and
            self.state[pos_end_r, pos_end_c] == player):
            return False  # Overline detected
            
        return True
    
    def check_winner(self, last_move_row, last_move_col):
        """
        Check if the last move resulted in a win.
        Checks all 4 directions for Standard Gomoku (5-only) wins.
        
        Args:
            last_move_row (int): Row of the last move
            last_move_col (int): Column of the last move
            
        Returns:
            int or None: Player (1 or -1) if win detected, None otherwise
        """
        if not (0 <= last_move_row < self.size and 0 <= last_move_col < self.size):
            return None
            
        player = self.state[last_move_row, last_move_col]
        if player == 0:  # No stone at this position
            return None
            
        # Check all 4 directions for wins
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical  
            (1, 1),   # Diagonal (↘)
            (1, -1),  # Anti-diagonal (↙)
        ]
        
        for dr, dc in directions:
            if self._check_line_win(last_move_row, last_move_col, player, dr, dc):
                return player
                
        return None
    
    def is_draw(self):
        """
        Check if the current board state is a draw.
        A draw occurs when the board is full (no legal moves) and no winner exists.
        
        Returns:
            bool: True if the game is a draw, False otherwise
        """
        # If there are still legal moves, it's not a draw
        if len(self.get_legal_moves()) > 0:
            return False
            
        # Board is full, check if there's a winner
        # Since we don't have a "last move" context, we need to check if any position
        # on the board results in a win condition
        for row in range(self.size):
            for col in range(self.size):
                if self.state[row, col] != 0:  # There's a stone here
                    if self.check_winner(row, col) is not None:
                        return False  # There's a winner, so not a draw
                        
        # Board is full and no winner found
        return True