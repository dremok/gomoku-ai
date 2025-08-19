"""
Self-play system for generating training data.

Plays games between agents and collects experience trajectories for training.
"""
import torch
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import json
from datetime import datetime

from ...core.game import Game
from ..agents.dqn_agent import DQNAgent
from ..agents.random_agent import RandomAgent
from ..agents.heuristic_agent import HeuristicAgent
from ..models.state_encoding import encode_game_state
from .replay_buffer import Transition, ReplayBuffer


class SelfPlayGame:
    """
    Manages a single self-play game and experience collection.
    
    Plays a complete game between two agents while recording all transitions
    for training purposes.
    """
    
    def __init__(self, 
                 player1_agent,
                 player2_agent,
                 collect_data: bool = True,
                 max_moves: int = 450,  # 15x15 board has 225 positions, allow for long games
                 verbose: bool = False):
        """
        Initialize self-play game.
        
        Args:
            player1_agent: Agent playing as Black (player 1)
            player2_agent: Agent playing as White (player -1)  
            collect_data: Whether to collect training data
            max_moves: Maximum moves before declaring draw
            verbose: Whether to print game progress
        """
        self.player1_agent = player1_agent
        self.player2_agent = player2_agent
        self.collect_data = collect_data
        self.max_moves = max_moves
        self.verbose = verbose
        
        # Game data collection
        self.game_trajectory = []
        self.game_states = []  # For analysis
        self.move_times = []
        
    def play_game(self) -> Dict[str, Any]:
        """
        Play a complete game and return results.
        
        Returns:
            dict: Game results including winner, moves, trajectory, etc.
        """
        game = Game()
        move_count = 0
        last_move = None
        
        # Track game start time
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting self-play game...")
            print(f"Player 1 (Black): {type(self.player1_agent).__name__}")
            print(f"Player 2 (White): {type(self.player2_agent).__name__}")
        
        # Game loop
        while game.game_state == 'ongoing' and move_count < self.max_moves:
            # Determine current agent
            if game.current_player == 1:  # Black
                current_agent = self.player1_agent
                agent_name = "Player1"
            else:  # White
                current_agent = self.player2_agent
                agent_name = "Player2"
            
            # Record state before move (for experience collection)
            if self.collect_data:
                pre_move_state = encode_game_state(game, last_move=last_move)
                self.game_states.append(pre_move_state.clone())
            
            # Get agent's move
            move_start = time.time()
            
            # Handle different agent interfaces
            if isinstance(current_agent, DQNAgent):
                move = current_agent.select_action(game, last_move=last_move)
            else:
                # RandomAgent and HeuristicAgent don't use last_move
                move = current_agent.select_action(game)
            move_time = time.time() - move_start
            self.move_times.append(move_time)
            
            if move is None:
                if self.verbose:
                    print(f"No valid moves available for {agent_name}")
                break
            
            if self.verbose:
                print(f"Move {move_count + 1}: {agent_name} plays {move}")
            
            # Make the move
            if not game.make_move(*move):
                if self.verbose:
                    print(f"Invalid move attempted: {move}")
                break
            
            # Collect experience data
            if self.collect_data:
                # We'll assign rewards after the game ends
                # For now, just store the transition structure
                experience = {
                    'move_number': move_count,
                    'player': game.current_player,  # Player who just moved
                    'state': pre_move_state,
                    'action': move,
                    'next_state': None,  # Will be filled in next iteration or at game end
                    'reward': 0.0,  # Will be assigned based on game outcome
                    'done': False   # Will be updated when game ends
                }
                self.game_trajectory.append(experience)
            
            last_move = move
            move_count += 1
        
        # Game finished - process results
        game_duration = time.time() - start_time
        
        # Determine final outcome
        if game.game_state == 'win':
            winner = game.winner
            outcome = 'win'
        elif move_count >= self.max_moves:
            winner = None
            outcome = 'draw_max_moves'
        else:
            winner = None
            outcome = 'draw_no_moves'
        
        # Process trajectory and assign rewards
        if self.collect_data and self.game_trajectory:
            self._assign_rewards_and_next_states(game, winner)
        
        # Compile results
        results = {
            'outcome': outcome,
            'winner': winner,
            'moves': move_count,
            'duration': game_duration,
            'avg_move_time': np.mean(self.move_times) if self.move_times else 0,
            'trajectory': self.game_trajectory if self.collect_data else [],
            'final_board': game.board.state.copy(),
            'player1_agent': type(self.player1_agent).__name__,
            'player2_agent': type(self.player2_agent).__name__
        }
        
        if self.verbose:
            print(f"Game finished: {outcome}")
            if winner:
                winner_name = "Player1 (Black)" if winner == 1 else "Player2 (White)"
                print(f"Winner: {winner_name}")
            print(f"Moves: {move_count}, Duration: {game_duration:.2f}s")
        
        return results
    
    def _assign_rewards_and_next_states(self, final_game: Game, winner: Optional[int]):
        """
        Assign rewards and next states to collected experiences.
        
        Args:
            final_game: Final game state
            winner: Winner of the game (1, -1, or None for draw)
        """
        if not self.game_trajectory:
            return
        
        # Assign next states (each state's next state is the following state)
        for i in range(len(self.game_trajectory) - 1):
            self.game_trajectory[i]['next_state'] = self.game_states[i + 1]
        
        # Last move has no next state (terminal)
        if self.game_trajectory:
            self.game_trajectory[-1]['next_state'] = None
            self.game_trajectory[-1]['done'] = True
        
        # Assign rewards based on game outcome
        for i, experience in enumerate(self.game_trajectory):
            player = experience['player']
            
            # Base reward for each move (small negative to encourage shorter games)
            reward = -0.01
            
            # Terminal state rewards
            if i == len(self.game_trajectory) - 1:  # Last move
                if winner == player:
                    reward = 1.0  # Win
                elif winner is None:
                    reward = 0.0  # Draw
                else:
                    reward = -1.0  # Loss
            
            experience['reward'] = reward
    
    def get_transitions(self) -> List[Transition]:
        """
        Convert game trajectory to list of Transition objects.
        
        Returns:
            List[Transition]: Transitions ready for replay buffer
        """
        transitions = []
        
        for exp in self.game_trajectory:
            # Convert action tuple to action index
            row, col = exp['action']
            action_idx = row * 15 + col  # Assuming 15x15 board
            
            transition = Transition(
                state=exp['state'],
                action=action_idx,
                reward=exp['reward'],
                next_state=exp['next_state'],
                done=exp.get('done', False)
            )
            transitions.append(transition)
        
        return transitions


class SelfPlayManager:
    """
    Manages multiple self-play games and data collection.
    
    Orchestrates self-play sessions, collects data, and provides utilities
    for training data generation.
    """
    
    def __init__(self, 
                 main_agent: DQNAgent,
                 opponent_agents: Optional[List] = None,
                 games_per_iteration: int = 100,
                 verbose: bool = True):
        """
        Initialize self-play manager.
        
        Args:
            main_agent: Main DQN agent being trained
            opponent_agents: List of opponent agents (includes main_agent, random, heuristic)
            games_per_iteration: Number of games to play per iteration
            verbose: Whether to print progress
        """
        self.main_agent = main_agent
        self.games_per_iteration = games_per_iteration
        self.verbose = verbose
        
        # Default opponent pool
        if opponent_agents is None:
            self.opponent_agents = [
                main_agent,  # Self-play
                RandomAgent(seed=42),
                HeuristicAgent(seed=42)
            ]
        else:
            self.opponent_agents = opponent_agents
        
        # Statistics tracking
        self.session_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_moves': 0,
            'total_duration': 0.0
        }
    
    def play_games(self, 
                   num_games: Optional[int] = None,
                   collect_data: bool = True) -> Tuple[List[Dict], List[Transition]]:
        """
        Play multiple self-play games.
        
        Args:
            num_games: Number of games to play (default: games_per_iteration)
            collect_data: Whether to collect training data
            
        Returns:
            tuple: (game_results, all_transitions)
        """
        if num_games is None:
            num_games = self.games_per_iteration
        
        all_results = []
        all_transitions = []
        
        if self.verbose:
            print(f"Starting {num_games} self-play games...")
        
        for game_idx in range(num_games):
            # Choose opponents (rotate through different opponents)
            player1 = self.main_agent
            player2 = self.opponent_agents[game_idx % len(self.opponent_agents)]
            
            # Randomly swap sides sometimes for balanced training
            if game_idx % 2 == 1:
                player1, player2 = player2, player1
            
            # Play the game
            self_play_game = SelfPlayGame(
                player1_agent=player1,
                player2_agent=player2,
                collect_data=collect_data,
                verbose=False  # Suppress individual game output
            )
            
            result = self_play_game.play_game()
            all_results.append(result)
            
            # Collect transitions
            if collect_data:
                transitions = self_play_game.get_transitions()
                all_transitions.extend(transitions)
            
            # Update statistics
            self._update_stats(result)
            
            # Progress reporting
            if self.verbose and (game_idx + 1) % 10 == 0:
                print(f"Completed {game_idx + 1}/{num_games} games")
        
        if self.verbose:
            self._print_session_summary()
        
        return all_results, all_transitions
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update session statistics with game result."""
        self.session_stats['games_played'] += 1
        self.session_stats['total_moves'] += result['moves']
        self.session_stats['total_duration'] += result['duration']
        
        if result['outcome'] == 'win':
            # Determine if main agent won
            if ((result['winner'] == 1 and result['player1_agent'] == 'DQNAgent') or
                (result['winner'] == -1 and result['player2_agent'] == 'DQNAgent')):
                self.session_stats['wins'] += 1
            else:
                self.session_stats['losses'] += 1
        else:
            self.session_stats['draws'] += 1
    
    def _print_session_summary(self):
        """Print summary of the self-play session."""
        stats = self.session_stats
        total_games = stats['games_played']
        
        if total_games == 0:
            return
        
        win_rate = stats['wins'] / total_games
        avg_moves = stats['total_moves'] / total_games
        avg_duration = stats['total_duration'] / total_games
        
        print(f"\n=== Self-Play Session Summary ===")
        print(f"Games played: {total_games}")
        print(f"Main agent win rate: {win_rate:.2%}")
        print(f"Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
        print(f"Average moves per game: {avg_moves:.1f}")
        print(f"Average game duration: {avg_duration:.2f}s")
        print(f"Total transitions collected: {stats['total_moves']}")
    
    def save_session_data(self, 
                         results: List[Dict], 
                         transitions: List[Transition],
                         save_dir: str):
        """
        Save self-play session data to disk.
        
        Args:
            results: Game results from play_games()
            transitions: Transitions from play_games()
            save_dir: Directory to save data
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save game results (JSON)
        results_file = save_dir / f"selfplay_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays, tensors and numpy scalars to Python types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types and tensors to Python types."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            serializable_results = convert_numpy_types(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save transitions to replay buffer
        buffer_file = save_dir / f"selfplay_buffer_{timestamp}.gz"
        replay_buffer = ReplayBuffer(capacity=len(transitions) + 1000)
        replay_buffer.add_trajectory(transitions)
        replay_buffer.save(buffer_file)
        
        # Save session statistics
        stats_file = save_dir / f"selfplay_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.session_stats, f, indent=2)
        
        if self.verbose:
            print(f"\nSession data saved to {save_dir}")
            print(f"Results: {results_file}")
            print(f"Replay buffer: {buffer_file}")
            print(f"Statistics: {stats_file}")
    
    def reset_stats(self):
        """Reset session statistics."""
        self.session_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_moves': 0,
            'total_duration': 0.0
        }