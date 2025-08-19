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