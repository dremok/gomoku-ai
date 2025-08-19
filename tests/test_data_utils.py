"""
Tests for training data utilities.
"""
import pytest
import torch
import tempfile
import os
import json
from pathlib import Path
from gomoku.ai.agents.dqn_agent import DQNAgent
from gomoku.ai.agents.random_agent import RandomAgent
from gomoku.ai.training.data_utils import (
    TrainingDataLoader, 
    TrainingDataGenerator, 
    TrainingBatchGenerator,
    create_training_dataset
)
from gomoku.ai.training.replay_buffer import ReplayBuffer, Transition
from gomoku.ai.training.self_play import SelfPlayManager


def create_dummy_replay_buffer(num_transitions=10):
    """Create a dummy replay buffer for testing."""
    buffer = ReplayBuffer(capacity=100)
    
    for i in range(num_transitions):
        state = torch.randn(4, 15, 15)
        next_state = torch.randn(4, 15, 15) if i < num_transitions - 1 else None
        done = i == num_transitions - 1
        
        buffer.add(
            state=state,
            action=i,
            reward=float(i),
            next_state=next_state,
            done=done
        )
    
    return buffer


def test_training_data_loader_initialization():
    """Test TrainingDataLoader initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = TrainingDataLoader(tmpdir)
        
        assert loader.data_dir == Path(tmpdir)
        assert loader.data_dir.exists()
        assert len(loader._replay_buffers) == 0
        assert len(loader._game_results) == 0


def test_training_data_loader_find_data_files():
    """Test finding data files in directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create some dummy files
        (tmpdir_path / "buffer_1.gz").touch()
        (tmpdir_path / "buffer_2.gz").touch()
        (tmpdir_path / "results_1.json").touch()
        (tmpdir_path / "stats_1.json").touch()
        (tmpdir_path / "other_file.txt").touch()
        
        loader = TrainingDataLoader(tmpdir)
        files = loader.find_data_files()
        
        assert len(files['replay_buffers']) == 2
        assert len(files['game_results']) == 1
        assert len(files['session_stats']) == 1
        
        # Check file names
        buffer_names = [f.name for f in files['replay_buffers']]
        assert 'buffer_1.gz' in buffer_names
        assert 'buffer_2.gz' in buffer_names


def test_training_data_loader_load_replay_buffer():
    """Test loading replay buffer from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save a buffer
        buffer = create_dummy_replay_buffer(5)
        buffer_path = os.path.join(tmpdir, 'test_buffer.gz')
        buffer.save(buffer_path)
        
        # Load using data loader
        loader = TrainingDataLoader(tmpdir)
        loaded_buffer = loader.load_replay_buffer('test_buffer.gz')
        
        assert len(loaded_buffer) == 5
        assert loaded_buffer.capacity == buffer.capacity
        
        # Test caching
        same_buffer = loader.load_replay_buffer('test_buffer.gz')
        assert same_buffer is loaded_buffer  # Should be same object from cache


def test_training_data_loader_load_game_results():
    """Test loading game results from JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy game results
        results = [
            {'outcome': 'win', 'winner': 1, 'moves': 10},
            {'outcome': 'draw', 'winner': None, 'moves': 15}
        ]
        
        results_path = os.path.join(tmpdir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        # Load using data loader
        loader = TrainingDataLoader(tmpdir)
        loaded_results = loader.load_game_results('test_results.json')
        
        assert len(loaded_results) == 2
        assert loaded_results[0]['outcome'] == 'win'
        assert loaded_results[1]['outcome'] == 'draw'
        
        # Test caching
        same_results = loader.load_game_results('test_results.json')
        assert same_results is loaded_results


def test_training_data_loader_load_nonexistent_file():
    """Test loading nonexistent files raises errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = TrainingDataLoader(tmpdir)
        
        with pytest.raises(FileNotFoundError):
            loader.load_replay_buffer('nonexistent.gz')
        
        with pytest.raises(FileNotFoundError):
            loader.load_game_results('nonexistent.json')


def test_training_data_loader_merge_replay_buffers():
    """Test merging multiple replay buffers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save multiple buffers
        buffer1 = create_dummy_replay_buffer(3)
        buffer2 = create_dummy_replay_buffer(4)
        
        buffer1_path = os.path.join(tmpdir, 'buffer1.gz')
        buffer2_path = os.path.join(tmpdir, 'buffer2.gz')
        
        buffer1.save(buffer1_path)
        buffer2.save(buffer2_path)
        
        # Merge buffers
        loader = TrainingDataLoader(tmpdir)
        merged_buffer = loader.merge_replay_buffers(
            ['buffer1.gz', 'buffer2.gz'],
            output_file='merged.gz'
        )
        
        # Check merged buffer
        assert len(merged_buffer) == 7  # 3 + 4
        assert merged_buffer.capacity >= 7
        
        # Check output file was created
        assert (Path(tmpdir) / 'merged.gz').exists()


def test_training_data_loader_merge_empty_list():
    """Test merging empty list raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = TrainingDataLoader(tmpdir)
        
        with pytest.raises(ValueError, match="No buffer files provided"):
            loader.merge_replay_buffers([])


def test_training_data_loader_get_data_statistics():
    """Test getting data statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test data
        buffer = create_dummy_replay_buffer(10)
        buffer_path = os.path.join(tmpdir, 'test_buffer.gz')
        buffer.save(buffer_path)
        
        results = [{'outcome': 'win', 'moves': 5}]
        results_path = os.path.join(tmpdir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        # Get statistics
        loader = TrainingDataLoader(tmpdir)
        stats = loader.get_data_statistics()
        
        assert 'data_dir' in stats
        assert stats['file_counts']['replay_buffers'] == 1
        assert stats['file_counts']['game_results'] == 1
        assert stats['total_transitions'] == 10
        assert stats['total_games'] == 1


def test_training_data_generator_initialization():
    """Test TrainingDataGenerator initializes correctly."""
    agent = DQNAgent(seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = TrainingDataGenerator(
            main_agent=agent,
            data_dir=tmpdir,
            games_per_session=50
        )
        
        assert generator.main_agent == agent
        assert generator.data_dir == Path(tmpdir)
        assert generator.games_per_session == 50
        assert isinstance(generator.self_play_manager, SelfPlayManager)


def test_training_data_generator_generate_training_data():
    """Test generating training data through self-play."""
    agent = DQNAgent(seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = TrainingDataGenerator(
            main_agent=agent,
            data_dir=tmpdir,
            games_per_session=3  # Small number for fast tests
        )
        
        # Generate training data
        generated_files = generator.generate_training_data(
            num_sessions=1,
            session_prefix="test"
        )
        
        # Check that files were generated
        assert len(generated_files) > 0
        
        # Check that session directory was created
        session_dirs = list(Path(tmpdir).glob("test_session_*"))
        assert len(session_dirs) == 1
        
        # Check that data files exist in session directory
        session_files = list(session_dirs[0].glob("*"))
        assert len(session_files) >= 3  # results, buffer, stats


def test_training_data_generator_update_opponent_pool():
    """Test updating opponent pool."""
    agent = DQNAgent(seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = TrainingDataGenerator(
            main_agent=agent,
            data_dir=tmpdir
        )
        
        initial_count = len(generator.self_play_manager.opponent_agents)
        
        # Add new opponents
        new_agents = [RandomAgent(seed=1), RandomAgent(seed=2)]
        generator.update_opponent_pool(new_agents)
        
        final_count = len(generator.self_play_manager.opponent_agents)
        assert final_count == initial_count + 2


def test_training_batch_generator_initialization():
    """Test TrainingBatchGenerator initializes correctly."""
    buffer = create_dummy_replay_buffer(20)
    
    generator = TrainingBatchGenerator(
        replay_buffer=buffer,
        batch_size=4,
        device='cpu'
    )
    
    assert generator.replay_buffer == buffer
    assert generator.batch_size == 4
    assert str(generator.device) == 'cpu'


def test_training_batch_generator_can_generate_batch():
    """Test checking if batch can be generated."""
    buffer = create_dummy_replay_buffer(10)
    
    generator = TrainingBatchGenerator(
        replay_buffer=buffer,
        batch_size=5
    )
    
    assert generator.can_generate_batch()
    
    # Test with batch size larger than buffer
    large_generator = TrainingBatchGenerator(
        replay_buffer=buffer,
        batch_size=15
    )
    
    assert not large_generator.can_generate_batch()


def test_training_batch_generator_generate_batch():
    """Test generating a single batch."""
    buffer = create_dummy_replay_buffer(20)
    
    generator = TrainingBatchGenerator(
        replay_buffer=buffer,
        batch_size=4
    )
    
    states, actions, rewards, next_states, dones = generator.generate_batch()
    
    assert states.shape == (4, 4, 15, 15)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 4, 15, 15)
    assert dones.shape == (4,)


def test_training_batch_generator_insufficient_data():
    """Test error when insufficient data for batch."""
    buffer = create_dummy_replay_buffer(2)
    
    generator = TrainingBatchGenerator(
        replay_buffer=buffer,
        batch_size=5
    )
    
    with pytest.raises(ValueError, match="Insufficient data"):
        generator.generate_batch()


def test_training_batch_generator_batch_iterator():
    """Test batch iterator functionality."""
    buffer = create_dummy_replay_buffer(20)
    
    generator = TrainingBatchGenerator(
        replay_buffer=buffer,
        batch_size=4
    )
    
    # Test limited iteration
    batches = list(generator.batch_iterator(num_batches=3))
    assert len(batches) == 3
    
    # Each batch should have correct shapes
    for states, actions, rewards, next_states, dones in batches:
        assert states.shape == (4, 4, 15, 15)
        assert actions.shape == (4,)


def test_training_batch_generator_get_data_statistics():
    """Test getting data statistics."""
    buffer = create_dummy_replay_buffer(20)
    
    generator = TrainingBatchGenerator(
        replay_buffer=buffer,
        batch_size=4
    )
    
    stats = generator.get_data_statistics()
    
    assert stats['buffer_size'] == 20
    assert stats['batch_size'] == 4
    assert stats['possible_batches'] == 5  # 20 // 4
    assert 'device' in stats


def test_create_training_dataset():
    """Test creating complete training dataset."""
    agent = DQNAgent(seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small dataset for testing
        buffer, batch_generator = create_training_dataset(
            data_dir=tmpdir,
            main_agent=agent,
            num_games=5,  # Very small for fast testing
            batch_size=2,
            device='cpu'
        )
        
        # Check buffer was created
        assert isinstance(buffer, ReplayBuffer)
        assert len(buffer) > 0
        
        # Check batch generator
        assert isinstance(batch_generator, TrainingBatchGenerator)
        assert batch_generator.batch_size == 2
        assert str(batch_generator.device) == 'cpu'
        
        # Check that we can generate batches
        if batch_generator.can_generate_batch():
            batch = batch_generator.generate_batch()
            assert len(batch) == 5  # states, actions, rewards, next_states, dones
        
        # Check that data files were created
        data_files = list(Path(tmpdir).glob("**/*"))
        assert len(data_files) > 0