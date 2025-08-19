"""
Training data preparation utilities.

Provides functions for loading, processing, and preparing training data
from self-play games and replay buffers.
"""
import torch
import numpy as np
import json
import gzip
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from datetime import datetime

from .replay_buffer import ReplayBuffer, Transition
from .self_play import SelfPlayManager
from ..agents.dqn_agent import DQNAgent


def get_default_data_dir() -> Path:
    """
    Get the default data directory for the project.
    
    Returns:
        Path: Default data directory (project_root/data)
    """
    # Get the project root (go up from gomoku/ai/training to project root)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


class TrainingDataLoader:
    """
    Loads and manages training data from various sources.
    
    Can load from:
    - Self-play session files
    - Replay buffer files
    - Individual game result files
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing training data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded data
        self._replay_buffers = {}
        self._game_results = {}
    
    def find_data_files(self) -> Dict[str, List[Path]]:
        """
        Find all training data files in the data directory (including subdirectories).
        
        Returns:
            dict: Mapping of file types to lists of file paths
        """
        files = {
            'replay_buffers': list(self.data_dir.glob('**/*buffer*.gz')),
            'game_results': list(self.data_dir.glob('**/*results*.json')),
            'session_stats': list(self.data_dir.glob('**/*stats*.json'))
        }
        
        return files
    
    def load_replay_buffer(self, buffer_file: str) -> ReplayBuffer:
        """
        Load a replay buffer from file.
        
        Args:
            buffer_file: Path to buffer file
            
        Returns:
            ReplayBuffer: Loaded replay buffer
        """
        buffer_path = Path(buffer_file)
        
        # Check cache first
        cache_key = str(buffer_path)
        if cache_key in self._replay_buffers:
            return self._replay_buffers[cache_key]
        
        # Load from file
        if not buffer_path.exists():
            # Try with data_dir prefix
            buffer_path = self.data_dir / buffer_file
        
        if not buffer_path.exists():
            raise FileNotFoundError(f"Replay buffer not found: {buffer_file}")
        
        buffer = ReplayBuffer.load_from_file(str(buffer_path))
        
        # Cache for future use
        self._replay_buffers[cache_key] = buffer
        
        return buffer
    
    def load_game_results(self, results_file: str) -> List[Dict]:
        """
        Load game results from JSON file.
        
        Args:
            results_file: Path to results file
            
        Returns:
            List[Dict]: List of game result dictionaries
        """
        results_path = Path(results_file)
        
        # Check cache first
        cache_key = str(results_path)
        if cache_key in self._game_results:
            return self._game_results[cache_key]
        
        # Try with data_dir prefix if not found
        if not results_path.exists():
            results_path = self.data_dir / results_file
        
        if not results_path.exists():
            raise FileNotFoundError(f"Game results not found: {results_file}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Cache for future use
        self._game_results[cache_key] = results
        
        return results
    
    def merge_replay_buffers(self, buffer_files: List[str], 
                           output_file: Optional[str] = None) -> ReplayBuffer:
        """
        Merge multiple replay buffers into one.
        
        Args:
            buffer_files: List of buffer file paths
            output_file: Optional path to save merged buffer
            
        Returns:
            ReplayBuffer: Merged replay buffer
        """
        if not buffer_files:
            raise ValueError("No buffer files provided")
        
        # Load first buffer to get capacity
        first_buffer = self.load_replay_buffer(buffer_files[0])
        
        # Calculate total capacity needed
        total_transitions = sum(len(self.load_replay_buffer(f)) for f in buffer_files)
        merged_capacity = max(first_buffer.capacity, total_transitions)
        
        # Create merged buffer
        merged_buffer = ReplayBuffer(
            capacity=merged_capacity,
            batch_size=first_buffer.batch_size
        )
        
        # Add all transitions
        for buffer_file in buffer_files:
            buffer = self.load_replay_buffer(buffer_file)
            
            # Extract all transitions
            if len(buffer) > 0:
                # Sample all transitions
                all_indices = list(range(len(buffer)))
                for idx in all_indices:
                    transition = buffer.buffer[idx]
                    merged_buffer.add(
                        transition.state,
                        transition.action,
                        transition.reward,
                        transition.next_state,
                        transition.done
                    )
        
        # Save if requested
        if output_file:
            output_path = self.data_dir / output_file
            merged_buffer.save(str(output_path))
        
        return merged_buffer
    
    def get_data_statistics(self) -> Dict:
        """
        Get statistics about available training data.
        
        Returns:
            dict: Statistics about data files and content
        """
        files = self.find_data_files()
        
        stats = {
            'data_dir': str(self.data_dir),
            'file_counts': {key: len(files_list) for key, files_list in files.items()},
            'replay_buffers': [],
            'total_transitions': 0,
            'total_games': 0
        }
        
        # Analyze replay buffers
        for buffer_file in files['replay_buffers']:
            try:
                buffer = self.load_replay_buffer(str(buffer_file))
                buffer_stats = buffer.get_stats()
                buffer_stats['file'] = buffer_file.name
                stats['replay_buffers'].append(buffer_stats)
                stats['total_transitions'] += buffer_stats['size']
            except Exception as e:
                print(f"Error loading buffer {buffer_file}: {e}")
        
        # Analyze game results
        for results_file in files['game_results']:
            try:
                results = self.load_game_results(str(results_file))
                stats['total_games'] += len(results)
            except Exception as e:
                print(f"Error loading results {results_file}: {e}")
        
        return stats


class TrainingDataGenerator:
    """
    Generates training data using self-play.
    
    Coordinates self-play sessions and data collection for training.
    """
    
    def __init__(self, 
                 main_agent: DQNAgent,
                 data_dir: str,
                 games_per_session: int = 100):
        """
        Initialize training data generator.
        
        Args:
            main_agent: Main DQN agent for self-play
            data_dir: Directory to save training data
            games_per_session: Number of games per self-play session
        """
        self.main_agent = main_agent
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.games_per_session = games_per_session
        
        # Initialize self-play manager
        self.self_play_manager = SelfPlayManager(
            main_agent=main_agent,
            games_per_iteration=games_per_session,
            verbose=True
        )
    
    def generate_training_data(self, 
                             num_sessions: int = 1,
                             session_prefix: str = "training") -> List[str]:
        """
        Generate training data through self-play sessions.
        
        Args:
            num_sessions: Number of self-play sessions to run
            session_prefix: Prefix for saved data files
            
        Returns:
            List[str]: List of generated data file paths
        """
        generated_files = []
        
        for session_idx in range(num_sessions):
            print(f"\n=== Training Data Generation Session {session_idx + 1}/{num_sessions} ===")
            
            # Reset stats for this session
            self.self_play_manager.reset_stats()
            
            # Play games and collect data
            results, transitions = self.self_play_manager.play_games(
                num_games=self.games_per_session,
                collect_data=True
            )
            
            # Save session data with session-specific naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = self.data_dir / f"{session_prefix}_session_{session_idx + 1}_{timestamp}"
            session_dir.mkdir(exist_ok=True)
            
            self.self_play_manager.save_session_data(results, transitions, str(session_dir))
            
            # Track generated files
            session_files = list(session_dir.glob("*"))
            generated_files.extend([str(f) for f in session_files])
            
            print(f"Session {session_idx + 1} completed. Generated {len(session_files)} files.")
        
        print(f"\n=== Training Data Generation Complete ===")
        print(f"Total sessions: {num_sessions}")
        print(f"Total files generated: {len(generated_files)}")
        
        return generated_files
    
    def update_opponent_pool(self, new_agents: List):
        """
        Update the opponent pool for self-play.
        
        Args:
            new_agents: List of new agents to add to opponent pool
        """
        current_opponents = self.self_play_manager.opponent_agents.copy()
        current_opponents.extend(new_agents)
        self.self_play_manager.opponent_agents = current_opponents
        
        print(f"Updated opponent pool. Now has {len(current_opponents)} opponents.")


class TrainingBatchGenerator:
    """
    Generates batches of training data for neural network training.
    
    Provides efficient batch iteration over replay buffers and handles
    data preprocessing for training.
    """
    
    def __init__(self, 
                 replay_buffer: ReplayBuffer,
                 batch_size: int = 32,
                 device: str = 'cpu'):
        """
        Initialize batch generator.
        
        Args:
            replay_buffer: Source of training data
            batch_size: Size of training batches
            device: Device to move data to
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.device = torch.device(device)
    
    def can_generate_batch(self) -> bool:
        """Check if buffer has enough data for a batch."""
        return self.replay_buffer.can_sample(self.batch_size)
    
    def generate_batch(self) -> Tuple[torch.Tensor, ...]:
        """
        Generate a single training batch.
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones) on specified device
        """
        if not self.can_generate_batch():
            raise ValueError(f"Insufficient data for batch size {self.batch_size}")
        
        # Sample from buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def batch_iterator(self, num_batches: Optional[int] = None) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Create an iterator over training batches.
        
        Args:
            num_batches: Number of batches to generate (None for infinite)
            
        Yields:
            tuple: Training batches
        """
        batch_count = 0
        
        while True:
            if num_batches is not None and batch_count >= num_batches:
                break
            
            if not self.can_generate_batch():
                break
            
            yield self.generate_batch()
            batch_count += 1
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about the training data."""
        return {
            'buffer_size': len(self.replay_buffer),
            'buffer_capacity': self.replay_buffer.capacity,
            'batch_size': self.batch_size,
            'possible_batches': len(self.replay_buffer) // self.batch_size,
            'device': str(self.device)
        }


def create_training_dataset(data_dir: Optional[str] = None,
                          main_agent: Optional[DQNAgent] = None,
                          num_games: int = 1000,
                          batch_size: int = 32,
                          device: str = 'cpu') -> Tuple[ReplayBuffer, TrainingBatchGenerator]:
    """
    Create a complete training dataset from scratch.
    
    Args:
        data_dir: Directory to save/load training data (default: project_root/data)
        main_agent: DQN agent for self-play (default: create new untrained agent)
        num_games: Number of self-play games to generate
        batch_size: Batch size for training
        device: Device for training batches
        
    Returns:
        tuple: (replay_buffer, batch_generator)
    """
    # Use defaults if not provided
    if data_dir is None:
        data_dir = str(get_default_data_dir())
    
    if main_agent is None:
        main_agent = DQNAgent(seed=42, epsilon=0.3)
    
    print(f"Creating training dataset with {num_games} games...")
    print(f"Data directory: {data_dir}")
    
    # Generate training data
    data_generator = TrainingDataGenerator(
        main_agent=main_agent,
        data_dir=data_dir,
        games_per_session=min(num_games, 100)  # Cap session size
    )
    
    num_sessions = (num_games + 99) // 100  # Ceiling division
    generated_files = data_generator.generate_training_data(
        num_sessions=num_sessions,
        session_prefix="dataset_creation"
    )
    
    # Load and merge all replay buffers
    data_loader = TrainingDataLoader(data_dir)
    files = data_loader.find_data_files()
    
    if not files['replay_buffers']:
        raise ValueError("No replay buffers were generated")
    
    # Merge all buffers
    merged_buffer = data_loader.merge_replay_buffers(
        [str(f) for f in files['replay_buffers']],
        output_file="merged_training_buffer.gz"
    )
    
    # Create batch generator
    batch_generator = TrainingBatchGenerator(
        replay_buffer=merged_buffer,
        batch_size=batch_size,
        device=device
    )
    
    print(f"Training dataset created successfully!")
    print(f"Buffer size: {len(merged_buffer)} transitions")
    print(f"Possible batches: {len(merged_buffer) // batch_size}")
    
    return merged_buffer, batch_generator