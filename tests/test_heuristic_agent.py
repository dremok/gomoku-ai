"""
Tests for HeuristicAgent class.
"""
import pytest
from gomoku.core.game import Game
from gomoku.ai.agents.heuristic_agent import HeuristicAgent


def test_heuristic_agent_initialization():
    """Test that HeuristicAgent initializes correctly."""
    # Test without seed
    agent = HeuristicAgent()
    assert agent.rng is not None
    
    # Test with seed for reproducibility
    agent_seeded = HeuristicAgent(seed=42)
    assert agent_seeded.rng is not None


def test_heuristic_agent_select_action_empty_board():
    """Test that HeuristicAgent selects valid moves on empty board."""
    game = Game()
    agent = HeuristicAgent(seed=42)
    
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


def test_heuristic_agent_prefers_center():
    """Test that HeuristicAgent prefers center moves on empty board."""
    game = Game()
    agent = HeuristicAgent(seed=42)
    
    moves = []
    for _ in range(10):  # Get multiple moves to see if they cluster near center
        move = agent.select_action(game)
        moves.append(move)
        
    # Check that moves are generally close to center (7, 7)
    center_distances = [((r - 7)**2 + (c - 7)**2)**0.5 for r, c in moves]
    avg_distance = sum(center_distances) / len(center_distances)
    
    # Should average closer than random (which would be ~5.7)
    assert avg_distance < 5.0


def test_heuristic_agent_immediate_win_detection():
    """Test that HeuristicAgent takes immediate wins."""
    game = Game()
    agent = HeuristicAgent()
    
    # Set up a position where black can win horizontally
    # Black stones at (7, 3), (7, 4), (7, 5), (7, 6) - can win at (7, 2) or (7, 7)
    for col in [3, 4, 5, 6]:
        game.board.apply_move(7, col, 1)  # Black stones
        
    # Set current player to black
    game.current_player = 1
    
    # Agent should choose a winning move
    move = agent.select_action(game)
    assert move in [(7, 2), (7, 7)], f"Expected winning move at (7,2) or (7,7), got {move}"
    
    # Verify the move actually creates a win
    game.board.apply_move(*move, 1)
    winner = game.board.check_winner(*move)
    assert winner == 1, "Move should result in a win for black"


def test_heuristic_agent_immediate_block_detection():
    """Test that HeuristicAgent blocks opponent wins."""
    game = Game()
    agent = HeuristicAgent()
    
    # Set up a position where white can win if not blocked
    # White stones at (7, 3), (7, 4), (7, 5), (7, 6) - would win at (7, 2) or (7, 7)
    for col in [3, 4, 5, 6]:
        game.board.apply_move(7, col, -1)  # White stones
        
    # Set current player to black (should block white's win)
    game.current_player = 1
    
    # Agent should block one of the winning moves
    move = agent.select_action(game)
    assert move in [(7, 2), (7, 7)], f"Expected blocking move at (7,2) or (7,7), got {move}"
    
    # Verify that without this move, white could win
    game.board.apply_move(*move, -1)  # Test if white could win at this position
    winner = game.board.check_winner(*move)
    game.board.state[move[0], move[1]] = 0  # Restore empty state
    assert winner == -1, "Position should be a winning position for white"


def test_heuristic_agent_win_over_block():
    """Test that HeuristicAgent prioritizes its own win over blocking."""
    game = Game()
    agent = HeuristicAgent()
    
    # Set up position where both players can win
    # Black can win at (5, 0) or (5, 5)
    for col in [1, 2, 3, 4]:
        game.board.apply_move(5, col, 1)  # Black stones
        
    # White can win at (7, 2) or (7, 7)  
    for col in [3, 4, 5, 6]:
        game.board.apply_move(7, col, -1)  # White stones
        
    # Set current player to black
    game.current_player = 1
    
    # Agent should take its own win rather than block white
    move = agent.select_action(game)
    black_win_moves = [(5, 0), (5, 5)]
    assert move in black_win_moves, f"Expected black to take its own win at {black_win_moves}, got {move}"
    
    # Verify the move creates a win for black
    game.board.apply_move(*move, 1)
    winner = game.board.check_winner(*move)
    assert winner == 1, "Move should result in a win for black"


def test_heuristic_agent_pattern_scoring():
    """Test that HeuristicAgent prefers creating strong patterns."""
    game = Game()
    agent = HeuristicAgent(seed=42)
    
    # Place some stones to create pattern scoring scenarios
    game.board.apply_move(7, 7, 1)  # Black center stone
    game.board.apply_move(6, 6, -1)  # White stone
    
    game.current_player = 1  # Black to move
    
    # Agent should prefer moves that extend the black stone pattern
    # Moves adjacent to (7, 7) should score higher
    move = agent.select_action(game)
    row, col = move
    
    # Should be within 2 squares of existing black stone
    distance_to_black = max(abs(row - 7), abs(col - 7))
    assert distance_to_black <= 2


def test_heuristic_agent_no_legal_moves():
    """Test that HeuristicAgent handles board with no legal moves."""
    game = Game()
    agent = HeuristicAgent()
    
    # Fill the entire board
    for row in range(15):
        for col in range(15):
            game.board.apply_move(row, col, 1 if (row + col) % 2 == 0 else -1)
    
    # Should return None when no moves available
    move = agent.select_action(game)
    assert move is None


def test_heuristic_agent_different_seeds():
    """Test that different seeds can produce different move sequences."""
    game1 = Game()
    game2 = Game()
    
    agent1 = HeuristicAgent(seed=1)
    agent2 = HeuristicAgent(seed=2)
    
    moves1 = []
    moves2 = []
    
    # Collect first 3 moves from each agent
    for _ in range(3):
        move1 = agent1.select_action(game1)
        move2 = agent2.select_action(game2)
        
        moves1.append(move1)
        moves2.append(move2)
        
        # Apply moves to keep boards synchronized
        if move1:
            game1.make_move(*move1)
        if move2:
            game2.make_move(*move2)
    
    # Different seeds might produce different sequences (though not guaranteed)
    # This test mainly ensures the agents can handle seeded randomness


def test_heuristic_agent_same_seed_reproducible():
    """Test that same seed produces reproducible results."""
    game1 = Game()
    game2 = Game()
    
    agent1 = HeuristicAgent(seed=42)
    agent2 = HeuristicAgent(seed=42)
    
    # First moves should be identical
    move1 = agent1.select_action(game1)
    move2 = agent2.select_action(game2)
    
    assert move1 == move2


def test_heuristic_agent_game_integration():
    """Test that HeuristicAgent can play a complete game with Game class."""
    game = Game()
    agent = HeuristicAgent(seed=42)
    
    moves_played = 0
    max_moves = 50  # Limit test to avoid very long games
    
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


def test_heuristic_agent_vertical_win_detection():
    """Test win detection in vertical direction."""
    game = Game()
    agent = HeuristicAgent()
    
    # Set up vertical win opportunity for black
    for row in [3, 4, 5, 6]:
        game.board.apply_move(row, 7, 1)  # Black stones
        
    game.current_player = 1
    
    # Agent should take the vertical win
    move = agent.select_action(game)
    assert move == (7, 7) or move == (2, 7)  # Either end completes the line


def test_heuristic_agent_diagonal_win_detection():
    """Test win detection in diagonal direction."""
    game = Game()
    agent = HeuristicAgent()
    
    # Set up diagonal win opportunity for black
    positions = [(3, 3), (4, 4), (5, 5), (6, 6)]
    for row, col in positions:
        game.board.apply_move(row, col, 1)  # Black stones
        
    game.current_player = 1
    
    # Agent should take the diagonal win
    move = agent.select_action(game)
    assert move == (7, 7) or move == (2, 2)  # Either end completes the diagonal


def test_heuristic_agent_overline_awareness():
    """Test that HeuristicAgent doesn't create overlines (6+ in a row)."""
    game = Game()
    agent = HeuristicAgent()
    
    # Set up a situation where placing a stone would create an overline
    # Black stones at positions that would create 6 in a row
    for col in [2, 3, 4, 5, 6, 8]:  # Gap at 7
        game.board.apply_move(7, col, 1)
        
    game.current_player = 1
    
    # Agent should NOT choose (7, 7) as it would create an overline
    move = agent.select_action(game)
    if move:  # If there are legal moves
        assert move != (7, 7), "Agent should not create overlines"
        
        # Verify that placing at (7, 7) would indeed create an invalid overline
        game.board.state[7, 7] = 1
        winner = game.board.check_winner(7, 7)
        game.board.state[7, 7] = 0
        assert winner is None, "Position (7,7) should not create a valid win (overline)"