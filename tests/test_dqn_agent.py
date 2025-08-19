"""
Tests for DQNAgent class.
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path
from gomoku.core.game import Game
from gomoku.ai.agents.dqn_agent import DQNAgent


def test_dqn_agent_initialization():
    """Test DQNAgent initializes correctly."""
    agent = DQNAgent()
    
    # Check default parameters
    assert agent.board_size == 15
    assert agent.epsilon == 0.1
    assert str(agent.device) == 'cpu'
    assert agent.rng is not None
    
    # Check model is created
    assert agent.model is not None
    assert not agent.is_trained
    assert agent.training_info['episodes'] == 0


def test_dqn_agent_custom_parameters():
    """Test DQNAgent with custom parameters."""
    agent = DQNAgent(board_size=9, epsilon=0.3, seed=42)
    
    assert agent.board_size == 9
    assert agent.epsilon == 0.3
    # Just check that rng is set (seed structure is internal implementation detail)
    assert agent.rng is not None


def test_dqn_agent_select_action_empty_board():
    """Test DQNAgent can select actions on empty board."""
    agent = DQNAgent(seed=42)
    game = Game()
    
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


def test_dqn_agent_select_action_partial_board():
    """Test DQNAgent selects only legal moves on partial board."""
    agent = DQNAgent(seed=42)
    game = Game()
    
    # Make some moves to reduce legal options
    game.make_move(7, 7)  # Black
    game.make_move(7, 8)  # White
    game.make_move(8, 7)  # Black
    
    # Agent should select a legal move
    move = agent.select_action(game, last_move=(8, 7))
    assert move is not None
    
    row, col = move
    assert game.board.state[row, col] == 0  # Should be empty
    assert (row, col) in game.board.get_legal_moves()


def test_dqn_agent_select_action_no_legal_moves():
    """Test DQNAgent handles board with no legal moves."""
    agent = DQNAgent()
    game = Game()
    
    # Fill the entire board
    for row in range(15):
        for col in range(15):
            game.board.apply_move(row, col, 1 if (row + col) % 2 == 0 else -1)
    
    # Should return None when no moves available
    move = agent.select_action(game)
    assert move is None


def test_dqn_agent_epsilon_setting():
    """Test epsilon setting and clamping."""
    agent = DQNAgent()
    
    # Test valid epsilon
    agent.set_epsilon(0.5)
    assert agent.epsilon == 0.5
    
    # Test clamping
    agent.set_epsilon(-0.1)
    assert agent.epsilon == 0.0
    
    agent.set_epsilon(1.5)
    assert agent.epsilon == 1.0


def test_dqn_agent_evaluation_mode():
    """Test evaluation mode setting."""
    agent = DQNAgent(epsilon=0.3)
    
    # Switch to evaluation mode
    agent.set_evaluation_mode(True)
    assert agent.epsilon == 0.0
    
    # Switch back to training mode
    agent.set_evaluation_mode(False)
    assert agent.epsilon == 0.3


def test_dqn_agent_get_q_values():
    """Test getting Q-values from agent."""
    agent = DQNAgent()
    game = Game()
    
    # Get Q-values
    q_values = agent.get_q_values(game)
    
    assert q_values.shape == (225,)
    assert q_values.dtype == torch.float32
    
    # Q-values should be reasonable (not all identical for untrained model)
    assert not torch.all(q_values == q_values[0])


def test_dqn_agent_game_integration():
    """Test DQNAgent can play a complete game."""
    agent = DQNAgent(seed=42)
    game = Game()
    
    moves_played = 0
    max_moves = 20  # Limit to avoid long games
    last_move = None
    
    while game.game_state == 'ongoing' and moves_played < max_moves:
        move = agent.select_action(game, last_move=last_move)
        assert move is not None, "Agent should find legal move while game ongoing"
        
        # Apply the move
        result = game.make_move(*move)
        assert result == True, f"Agent move {move} should be legal"
        
        last_move = move
        moves_played += 1
    
    # Game should have made progress
    assert moves_played > 0
    assert len(game.board.get_legal_moves()) < 225  # Some moves were made


def test_dqn_agent_save_load():
    """Test saving and loading DQN agent."""
    agent = DQNAgent(epsilon=0.2, seed=42)
    
    # Modify some training info
    agent.is_trained = True
    agent.training_info['episodes'] = 100
    agent.training_info['steps'] = 5000
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save agent
        model_path = os.path.join(tmpdir, 'test_model.pt')
        agent.save(model_path, metadata={'test': 'value'})
        
        # Check files were created
        assert os.path.exists(model_path)
        assert os.path.exists(os.path.join(tmpdir, 'test_model.json'))
        
        # Create new agent and load
        new_agent = DQNAgent(epsilon=0.5)  # Different epsilon
        new_agent.load(model_path)
        
        # Check loaded values
        assert new_agent.is_trained == True
        assert new_agent.training_info['episodes'] == 100
        assert new_agent.training_info['steps'] == 5000
        
        # Models should have same weights
        for (name1, param1), (name2, param2) in zip(
            agent.model.named_parameters(), 
            new_agent.model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


def test_dqn_agent_load_from_file():
    """Test loading agent from file (class method)."""
    # Create and save an agent
    agent = DQNAgent(epsilon=0.1, seed=42)
    agent.is_trained = True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.pt')
        agent.save(model_path)
        
        # Load using class method
        loaded_agent = DQNAgent.load_from_file(model_path, epsilon=0.4)
        
        # Check parameters
        assert loaded_agent.board_size == 15
        assert loaded_agent.epsilon == 0.4  # Should use provided epsilon
        assert loaded_agent.is_trained == True


def test_dqn_agent_load_nonexistent_file():
    """Test loading from nonexistent file raises error."""
    agent = DQNAgent()
    
    with pytest.raises(FileNotFoundError):
        agent.load('nonexistent_model.pt')


def test_dqn_agent_load_incompatible_board_size():
    """Test loading model with different board size raises error."""
    # Create agent with board size 9
    agent_9x9 = DQNAgent(board_size=9)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model_9x9.pt')
        agent_9x9.save(model_path)
        
        # Try to load into 15x15 agent
        agent_15x15 = DQNAgent(board_size=15)
        with pytest.raises(ValueError, match="board size"):
            agent_15x15.load(model_path)


def test_dqn_agent_get_info():
    """Test getting agent information."""
    agent = DQNAgent(epsilon=0.25)
    
    info = agent.get_info()
    
    assert info['type'] == 'DQNAgent'
    assert info['board_size'] == 15
    assert info['epsilon'] == 0.25
    assert info['is_trained'] == False
    assert 'model_parameters' in info
    assert info['model_parameters'] > 0


def test_dqn_agent_string_representation():
    """Test string representation of agent."""
    agent = DQNAgent(epsilon=0.15)
    
    str_repr = str(agent)
    assert 'DQNAgent' in str_repr
    assert '0.15' in str_repr
    assert 'untrained' in str_repr
    assert 'params' in str_repr


def test_dqn_agent_reproducible_behavior():
    """Test that same seed produces reproducible behavior for action selection."""
    game = Game()
    game.make_move(7, 7)  # Make initial move
    
    # Create two agents with same seed and force fully random exploration
    agent1 = DQNAgent(seed=42, epsilon=1.0)  # Fully random = only RNG affects choices
    agent2 = DQNAgent(seed=42, epsilon=1.0)
    
    # With fully random policy, same seed should produce identical moves
    moves1 = []
    moves2 = []
    
    test_game1 = Game()
    test_game2 = Game()
    test_game1.make_move(7, 7)  # Same initial state
    test_game2.make_move(7, 7)
    
    for _ in range(3):
        move1 = agent1.select_action(test_game1)
        move2 = agent2.select_action(test_game2)
        moves1.append(move1)
        moves2.append(move2)
    
    # With ε=1.0 (fully random) and same seed, should make identical moves
    assert moves1 == moves2, "Agents with same seed and ε=1.0 should make identical moves"


def test_dqn_agent_different_seeds():
    """Test that different seeds can produce different behavior."""
    game = Game()
    
    agent1 = DQNAgent(seed=1)
    agent2 = DQNAgent(seed=2)
    
    moves1 = []
    moves2 = []
    
    # Get several moves
    for _ in range(10):
        move1 = agent1.select_action(game)
        move2 = agent2.select_action(game)
        moves1.append(move1)
        moves2.append(move2)
    
    # Different seeds should produce some different moves
    # (Though untrained networks might still be similar)
    different_moves = sum(1 for m1, m2 in zip(moves1, moves2) if m1 != m2)
    # Allow for some similarity in untrained networks
    assert different_moves >= 1, "Different seeds should produce at least some different moves"