#!/usr/bin/env python3
"""
Simple evaluation script for testing agents against each other.
"""
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gomoku.core.game import Game
from gomoku.ai.agents.random_agent import RandomAgent  
from gomoku.ai.agents.heuristic_agent import HeuristicAgent


def play_game(agent1, agent2, show_progress=False):
    """
    Play a single game between two agents.
    
    Args:
        agent1: Agent playing black (goes first)
        agent2: Agent playing white  
        show_progress: Whether to print move-by-move progress
        
    Returns:
        int: Winner (1 for agent1/black, -1 for agent2/white, 0 for draw)
    """
    game = Game()
    move_count = 0
    
    while game.game_state == 'ongoing' and move_count < 225:  # Max possible moves
        current_agent = agent1 if game.current_player == 1 else agent2
        
        move = current_agent.select_action(game)
        if move is None:
            break  # No legal moves (shouldn't happen)
            
        if show_progress:
            print(f"Move {move_count + 1}: Player {game.current_player} plays {move}")
            
        success = game.make_move(*move)
        if not success:
            print(f"ERROR: Invalid move {move} by player {game.current_player}")
            break
            
        move_count += 1
        
    if show_progress:
        print(f"Game ended after {move_count} moves: {game.game_state}")
        if game.game_state == 'win':
            print(f"Winner: Player {game.winner}")
            
    if game.game_state == 'win':
        return game.winner
    else:
        return 0  # Draw


def evaluate_agents(agent1_name, agent1, agent2_name, agent2, num_games=100, swap_colors=True):
    """
    Evaluate two agents by playing multiple games.
    
    Args:
        agent1_name: Name of agent1 for display
        agent1: Agent1 instance
        agent2_name: Name of agent2 for display  
        agent2: Agent2 instance
        num_games: Number of games to play
        swap_colors: Whether to swap colors every game
        
    Returns:
        dict: Results summary
    """
    results = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'agent1_as_black_wins': 0,
        'agent1_as_white_wins': 0,
        'agent2_as_black_wins': 0,
        'agent2_as_white_wins': 0,
        'games_as_black': 0,
        'games_as_white': 0
    }
    
    print(f"Evaluating {agent1_name} vs {agent2_name}")
    print(f"Playing {num_games} games{'with color swapping' if swap_colors else ''}...")
    print()
    
    start_time = time.time()
    
    for i in range(num_games):
        if swap_colors and i % 2 == 1:
            # Swap colors - agent2 goes first (black), agent1 second (white)
            winner = play_game(agent2, agent1)
            results['games_as_white'] += 1
            
            if winner == 1:  # Black won (agent2)
                results['agent2_wins'] += 1
                results['agent2_as_black_wins'] += 1
            elif winner == -1:  # White won (agent1) 
                results['agent1_wins'] += 1
                results['agent1_as_white_wins'] += 1
            else:
                results['draws'] += 1
        else:
            # Normal colors - agent1 goes first (black), agent2 second (white)
            winner = play_game(agent1, agent2)
            results['games_as_black'] += 1
            
            if winner == 1:  # Black won (agent1)
                results['agent1_wins'] += 1
                results['agent1_as_black_wins'] += 1
            elif winner == -1:  # White won (agent2)
                results['agent2_wins'] += 1
                results['agent2_as_white_wins'] += 1
            else:
                results['draws'] += 1
                
        # Show progress
        if (i + 1) % max(1, num_games // 20) == 0:
            progress = (i + 1) / num_games * 100
            print(f"Progress: {progress:.0f}% ({i + 1}/{num_games})")
    
    elapsed = time.time() - start_time
    
    # Calculate win percentages
    total_games = results['agent1_wins'] + results['agent2_wins'] + results['draws']
    agent1_win_rate = results['agent1_wins'] / total_games * 100 if total_games > 0 else 0
    agent2_win_rate = results['agent2_wins'] / total_games * 100 if total_games > 0 else 0
    draw_rate = results['draws'] / total_games * 100 if total_games > 0 else 0
    
    # Print results
    print(f"\n=== Results after {total_games} games ({elapsed:.1f}s) ===")
    print(f"{agent1_name}: {results['agent1_wins']} wins ({agent1_win_rate:.1f}%)")
    print(f"{agent2_name}: {results['agent2_wins']} wins ({agent2_win_rate:.1f}%)")
    print(f"Draws: {results['draws']} ({draw_rate:.1f}%)")
    print()
    
    if swap_colors:
        print("=== Color Balance ===")
        print(f"{agent1_name} as Black: {results['agent1_as_black_wins']}/{results['games_as_black']} wins")
        print(f"{agent1_name} as White: {results['agent1_as_white_wins']}/{results['games_as_white']} wins")
        print(f"{agent2_name} as Black: {results['agent2_as_black_wins']}/{results['games_as_black']} wins")
        print(f"{agent2_name} as White: {results['agent2_as_white_wins']}/{results['games_as_white']} wins")
        print()
    
    results.update({
        'total_games': total_games,
        'agent1_win_rate': agent1_win_rate,
        'agent2_win_rate': agent2_win_rate,
        'draw_rate': draw_rate,
        'elapsed_time': elapsed
    })
    
    return results


def main():
    """Main evaluation function."""
    print("Gomoku Agent Evaluation")
    print("======================")
    print()
    
    # Test HeuristicAgent vs RandomAgent
    heuristic_agent = HeuristicAgent(seed=42)
    random_agent = RandomAgent(seed=123)
    
    results = evaluate_agents(
        "HeuristicAgent", heuristic_agent,
        "RandomAgent", random_agent,
        num_games=200,  # Test with 200 games for statistical significance
        swap_colors=True
    )
    
    # Check if HeuristicAgent meets the >80% win rate requirement
    heuristic_win_rate = results['agent1_win_rate']
    print("=== Acceptance Criteria ===")
    print(f"HeuristicAgent win rate: {heuristic_win_rate:.1f}%")
    print(f"Required: >80%")
    
    if heuristic_win_rate > 80:
        print("✅ PASS: HeuristicAgent > RandomAgent by required margin")
    else:
        print("❌ FAIL: HeuristicAgent did not meet >80% win rate requirement")
    
    return results


if __name__ == "__main__":
    main()