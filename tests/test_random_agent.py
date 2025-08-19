"""
Tests for RandomAgent class.
"""
import pytest
from gomoku.core.game import Game
from gomoku.ai.agents.random_agent import RandomAgent


def test_random_agent_initialization():
    """Test that RandomAgent initializes correctly."""
    # Test without seed
    agent = RandomAgent()
    assert agent.rng is not None
    
    # Test with seed for reproducibility
    agent_seeded = RandomAgent(seed=42)
    assert agent_seeded.rng is not None


def test_random_agent_select_action_empty_board():
    """Test that RandomAgent selects valid moves on empty board."""
    game = Game()
    agent = RandomAgent(seed=42)  # Use seed for reproducible tests
    
    # Should select a valid move
    move = agent.select_action(game)
    assert move is not None
    assert isinstance(move, tuple)
    assert len(move) == 2
    
    row, col = move
    assert 0 <= row < 15
    assert 0 <= col < 15
    
    # Move should be legal (empty position)
    assert game.board.state[row, col] == 0


def test_random_agent_select_action_partial_board():
    """Test that RandomAgent only selects from legal moves on partial board."""
    game = Game()
    agent = RandomAgent(seed=42)
    
    # Place some stones to reduce legal moves
    game.make_move(7, 7)  # Black
    game.make_move(7, 8)  # White
    game.make_move(8, 7)  # Black
    
    # Agent should only select from remaining legal moves
    move = agent.select_action(game)
    assert move is not None
    
    row, col = move
    assert game.board.state[row, col] == 0  # Should be empty
    assert (row, col) in game.board.get_legal_moves()


def test_random_agent_no_legal_moves():
    """Test that RandomAgent handles board with no legal moves."""
    game = Game()
    agent = RandomAgent()
    
    # Fill the entire board
    for row in range(15):
        for col in range(15):
            game.board.apply_move(row, col, 1 if (row + col) % 2 == 0 else -1)
    
    # Should return None when no moves available
    move = agent.select_action(game)
    assert move is None


def test_random_agent_different_seeds():
    """Test that different seeds produce different move sequences."""
    game1 = Game()
    game2 = Game()
    
    agent1 = RandomAgent(seed=1)
    agent2 = RandomAgent(seed=2)
    
    moves1 = []
    moves2 = []
    
    # Collect first 5 moves from each agent
    for _ in range(5):
        move1 = agent1.select_action(game1)
        move2 = agent2.select_action(game2)
        
        moves1.append(move1)
        moves2.append(move2)
        
        # Apply moves to keep boards synchronized
        if move1:
            game1.make_move(*move1)
        if move2:
            game2.make_move(*move2)
    
    # Different seeds should produce different sequences
    assert moves1 != moves2


def test_random_agent_same_seed_reproducible():
    """Test that same seed produces reproducible results."""
    game1 = Game()
    game2 = Game()
    
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=42)
    
    # First moves should be identical
    move1 = agent1.select_action(game1)
    move2 = agent2.select_action(game2)
    
    assert move1 == move2


def test_random_agent_game_integration():
    """Test that RandomAgent can play a complete game with Game class."""
    game = Game()
    agent = RandomAgent(seed=42)
    
    moves_played = 0
    max_moves = 20  # Limit test to avoid infinite games
    
    while game.game_state == 'ongoing' and moves_played < max_moves:
        move = agent.select_action(game)
        assert move is not None, "Agent should find legal move while game ongoing"
        
        # Apply the move
        result = game.make_move(*move)
        assert result == True, f"Agent move {move} should be legal"
        
        moves_played += 1
    
    # Game should have made progress
    assert moves_played > 0
    assert len(game.board.get_legal_moves()) < 225  # Some moves were made
    
    # If game ended, it should be in a valid end state
    if game.game_state != 'ongoing':
        assert game.game_state in ['win', 'draw']