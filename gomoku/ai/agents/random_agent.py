"""
Random agent for Gomoku.
"""
import random


class RandomAgent:
    """
    An agent that plays random legal moves.
    
    This is the simplest possible agent - it just selects uniformly
    at random from all available legal moves.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the random agent.
        
        Args:
            seed (int, optional): Random seed for reproducible behavior
        """
        self.rng = random.Random(seed)
        
    def select_action(self, game):
        """
        Select a random legal move from the current game state.
        
        Args:
            game: Game instance with current board state
            
        Returns:
            tuple: (row, col) coordinates of selected move, or None if no legal moves
        """
        legal_moves = game.board.get_legal_moves()
        
        if not legal_moves:
            return None
            
        return self.rng.choice(legal_moves)