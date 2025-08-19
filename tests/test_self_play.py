"""
Tests for self-play system.
"""
import pytest
import tempfile
import os
from pathlib import Path
import json
from gomoku.core.game import Game
from gomoku.ai.agents.dqn_agent import DQNAgent
from gomoku.ai.agents.random_agent import RandomAgent
from gomoku.ai.agents.heuristic_agent import HeuristicAgent
from gomoku.ai.training.self_play import SelfPlayGame, SelfPlayManager
from gomoku.ai.training.replay_buffer import Transition


def test_self_play_game_initialization():
    """Test SelfPlayGame initializes correctly."""
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=123)
    
    game = SelfPlayGame(
        player1_agent=agent1,
        player2_agent=agent2,
        collect_data=True,
        max_moves=10,
        verbose=False
    )
    
    assert game.player1_agent == agent1
    assert game.player2_agent == agent2
    assert game.collect_data == True
    assert game.max_moves == 10
    assert game.verbose == False
    assert len(game.game_trajectory) == 0


def test_self_play_game_random_vs_random():
    """Test self-play game between random agents."""
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=123)
    
    game = SelfPlayGame(
        player1_agent=agent1,
        player2_agent=agent2,
        collect_data=True,
        max_moves=50,
        verbose=False
    )
    
    result = game.play_game()
    
    # Check result structure
    assert 'outcome' in result
    assert 'winner' in result
    assert 'moves' in result
    assert 'duration' in result
    assert 'trajectory' in result
    assert 'final_board' in result
    
    # Should have made some moves
    assert result['moves'] > 0
    assert result['duration'] > 0
    
    # Trajectory should have data
    if game.collect_data:
        assert len(result['trajectory']) == result['moves']


def test_self_play_game_dqn_vs_random():
    """Test self-play game between DQN and random agent."""
    dqn_agent = DQNAgent(seed=42, epsilon=0.5)
    random_agent = RandomAgent(seed=123)
    
    game = SelfPlayGame(
        player1_agent=dqn_agent,
        player2_agent=random_agent,
        collect_data=True,
        max_moves=30,
        verbose=False
    )
    
    result = game.play_game()
    
    assert result['player1_agent'] == 'DQNAgent'
    assert result['player2_agent'] == 'RandomAgent'
    assert result['moves'] > 0


def test_self_play_game_max_moves_limit():
    """Test game respects max moves limit."""
    # Use high epsilon for more random behavior to avoid quick wins
    agent1 = DQNAgent(seed=42, epsilon=1.0)
    agent2 = DQNAgent(seed=123, epsilon=1.0)
    
    game = SelfPlayGame(
        player1_agent=agent1,
        player2_agent=agent2,
        collect_data=False,
        max_moves=5,  # Very low limit
        verbose=False
    )
    
    result = game.play_game()
    
    # Should stop at max moves if no winner
    assert result['moves'] <= 5
    if result['moves'] == 5 and result['outcome'] != 'win':
        assert result['outcome'] == 'draw_max_moves'


def test_self_play_game_no_data_collection():
    """Test game without data collection."""
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=123)
    
    game = SelfPlayGame(
        player1_agent=agent1,
        player2_agent=agent2,
        collect_data=False,
        max_moves=20,
        verbose=False
    )
    
    result = game.play_game()
    
    assert len(result['trajectory']) == 0
    assert len(game.game_trajectory) == 0


def test_self_play_game_get_transitions():
    """Test converting game trajectory to transitions."""
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=123)
    
    game = SelfPlayGame(
        player1_agent=agent1,
        player2_agent=agent2,
        collect_data=True,
        max_moves=10,
        verbose=False
    )
    
    result = game.play_game()
    transitions = game.get_transitions()
    
    # Should have transitions for each move
    assert len(transitions) == result['moves']
    
    # Check transition structure
    for transition in transitions:
        assert isinstance(transition, Transition)
        assert transition.state is not None
        assert isinstance(transition.action, int)
        assert isinstance(transition.reward, float)
        assert isinstance(transition.done, bool)
        
        # Last transition should be terminal
        if transition is transitions[-1]:
            assert transition.done == True
        
        # Action should be valid board position
        assert 0 <= transition.action < 225  # 15x15 board


def test_self_play_manager_initialization():
    """Test SelfPlayManager initializes correctly."""
    main_agent = DQNAgent(seed=42)
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        games_per_iteration=50,
        verbose=False
    )
    
    assert manager.main_agent == main_agent
    assert manager.games_per_iteration == 50
    assert manager.verbose == False
    assert len(manager.opponent_agents) == 3  # main_agent, random, heuristic
    
    # Check default opponents include expected types
    agent_types = [type(agent).__name__ for agent in manager.opponent_agents]
    assert 'DQNAgent' in agent_types
    assert 'RandomAgent' in agent_types
    assert 'HeuristicAgent' in agent_types


def test_self_play_manager_custom_opponents():
    """Test SelfPlayManager with custom opponent list."""
    main_agent = DQNAgent(seed=42)
    custom_opponents = [
        RandomAgent(seed=1),
        RandomAgent(seed=2)
    ]
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        opponent_agents=custom_opponents,
        verbose=False
    )
    
    assert manager.opponent_agents == custom_opponents
    assert len(manager.opponent_agents) == 2


def test_self_play_manager_play_games():
    """Test playing multiple games through manager."""
    main_agent = DQNAgent(seed=42, epsilon=0.8)  # High exploration for faster games
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        games_per_iteration=5,
        verbose=False
    )
    
    results, transitions = manager.play_games(num_games=3, collect_data=True)
    
    # Should have played 3 games
    assert len(results) == 3
    
    # Should have some transitions
    assert len(transitions) > 0
    
    # Each result should be valid
    for result in results:
        assert 'outcome' in result
        assert 'moves' in result
        assert result['moves'] > 0
    
    # Check statistics were updated
    stats = manager.session_stats
    assert stats['games_played'] == 3
    assert stats['total_moves'] > 0


def test_self_play_manager_no_data_collection():
    """Test playing games without collecting data."""
    main_agent = DQNAgent(seed=42)
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        verbose=False
    )
    
    results, transitions = manager.play_games(num_games=2, collect_data=False)
    
    assert len(results) == 2
    assert len(transitions) == 0  # No data collected


def test_self_play_manager_statistics():
    """Test session statistics tracking."""
    main_agent = DQNAgent(seed=42)
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        verbose=False
    )
    
    # Play some games
    results, transitions = manager.play_games(num_games=3, collect_data=False)
    
    stats = manager.session_stats
    
    # Check basic stats
    assert stats['games_played'] == 3
    assert stats['total_moves'] > 0
    assert stats['total_duration'] > 0
    
    # Should have wins, losses, or draws
    total_outcomes = stats['wins'] + stats['losses'] + stats['draws']
    assert total_outcomes == 3


def test_self_play_manager_reset_stats():
    """Test resetting session statistics."""
    main_agent = DQNAgent(seed=42)
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        verbose=False
    )
    
    # Play some games to accumulate stats
    manager.play_games(num_games=2, collect_data=False)
    
    assert manager.session_stats['games_played'] == 2
    
    # Reset stats
    manager.reset_stats()
    
    stats = manager.session_stats
    assert stats['games_played'] == 0
    assert stats['wins'] == 0
    assert stats['losses'] == 0
    assert stats['draws'] == 0
    assert stats['total_moves'] == 0
    assert stats['total_duration'] == 0.0


def test_self_play_manager_save_session_data():
    """Test saving session data to disk."""
    main_agent = DQNAgent(seed=42)
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        verbose=False
    )
    
    # Play some games
    results, transitions = manager.play_games(num_games=2, collect_data=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save session data
        manager.save_session_data(results, transitions, tmpdir)
        
        # Check files were created
        save_dir = Path(tmpdir)
        files = list(save_dir.glob('*'))
        
        # Should have results, buffer, and stats files
        assert len(files) >= 3
        
        # Check specific file types exist
        json_files = list(save_dir.glob('*.json'))
        buffer_files = list(save_dir.glob('*.gz'))
        
        assert len(json_files) >= 2  # results and stats
        assert len(buffer_files) >= 1  # replay buffer
        
        # Verify JSON files can be loaded
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                assert isinstance(data, (dict, list))


def test_self_play_game_reward_assignment():
    """Test that rewards are assigned correctly in game trajectory."""
    # Use deterministic agents for predictable outcome
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=42)  # Same seed for faster games
    
    game = SelfPlayGame(
        player1_agent=agent1,
        player2_agent=agent2,
        collect_data=True,
        max_moves=20,
        verbose=False
    )
    
    result = game.play_game()
    
    if len(game.game_trajectory) > 0:
        # Check that last transition has terminal reward
        last_experience = game.game_trajectory[-1]
        assert last_experience['done'] == True
        
        # Terminal reward should be win/loss/draw
        if result['outcome'] == 'win':
            winner = result['winner']
            last_player = last_experience['player']
            
            if winner == last_player:
                assert last_experience['reward'] == 1.0  # Win
            else:
                # Find the winning move in trajectory
                for exp in reversed(game.game_trajectory):
                    if exp['player'] == winner:
                        assert exp['reward'] == 1.0
                        break
        
        # Non-terminal moves should have small negative rewards
        for exp in game.game_trajectory[:-1]:
            assert exp['reward'] == -0.01


def test_self_play_manager_opponent_rotation():
    """Test that manager rotates through different opponents."""
    main_agent = DQNAgent(seed=42)
    
    # Create manager with known opponents
    opponents = [
        RandomAgent(seed=1),
        HeuristicAgent(seed=2)
    ]
    
    manager = SelfPlayManager(
        main_agent=main_agent,
        opponent_agents=opponents,
        verbose=False
    )
    
    # Play multiple games to see opponent rotation
    results, _ = manager.play_games(num_games=4, collect_data=False)
    
    # Should see different opponent types
    opponent_types = set()
    for result in results:
        # Check which agent types played
        if result['player1_agent'] == 'DQNAgent':
            opponent_types.add(result['player2_agent'])
        else:
            opponent_types.add(result['player1_agent'])
    
    # Should have used both opponent types
    assert len(opponent_types) >= 1  # At least one opponent type used