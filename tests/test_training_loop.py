"""
Tests for training loop.
"""
import pytest
import tempfile
import json
import os
from pathlib import Path
from gomoku.ai.training.training_loop import TrainingConfig, TrainingLoop


def test_training_config_creation():
    """Test training configuration creation."""
    config = TrainingConfig(
        total_steps=1000,
        batch_size=16,
        learning_rate=1e-4
    )
    
    assert config.total_steps == 1000
    assert config.batch_size == 16
    assert config.learning_rate == 1e-4
    assert config.gamma == 0.99  # Default value
    
    # Test conversion to/from dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict['total_steps'] == 1000
    
    new_config = TrainingConfig.from_dict(config_dict)
    assert new_config.total_steps == 1000
    assert new_config.batch_size == 16


def test_training_loop_initialization():
    """Test training loop initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            total_steps=100,
            batch_size=4,
            games_per_update=2,
            min_replay_size=10
        )
        
        training_loop = TrainingLoop(
            config=config,
            data_dir=tmpdir,
            run_name="test_run"
        )
        
        # Check initialization
        assert training_loop.config == config
        assert training_loop.step == 0
        assert training_loop.episode == 0
        
        # Check directory structure
        assert training_loop.run_dir.exists()
        assert training_loop.models_dir.exists()
        assert training_loop.logs_dir.exists()
        
        # Check components
        assert training_loop.main_agent is not None
        assert training_loop.trainer is not None
        assert training_loop.replay_buffer is not None
        assert training_loop.self_play_manager is not None
        
        # Check config was saved
        config_file = training_loop.run_dir / "config.json"
        assert config_file.exists()
        
        with open(config_file) as f:
            saved_config = json.load(f)
        assert saved_config['total_steps'] == 100


def test_training_loop_epsilon_decay():
    """Test epsilon decay functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            self_play_epsilon=0.5,
            epsilon_min=0.1,
            epsilon_decay_steps=100
        )
        
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        # Initial epsilon
        initial_epsilon = training_loop.main_agent.epsilon
        assert initial_epsilon == 0.5
        
        # Simulate training steps
        training_loop.step = 50  # Halfway through decay
        mid_epsilon = training_loop._update_epsilon()
        assert 0.1 < mid_epsilon < 0.5
        
        # End of decay
        training_loop.step = 100
        final_epsilon = training_loop._update_epsilon()
        assert final_epsilon == 0.1
        
        # Beyond decay
        training_loop.step = 150
        post_epsilon = training_loop._update_epsilon()
        assert post_epsilon == 0.1


def test_training_loop_self_play_data_generation():
    """Test self-play data generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            games_per_update=2,
            batch_size=4
        )
        
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        # Generate data
        initial_buffer_size = len(training_loop.replay_buffer)
        new_transitions = training_loop._generate_self_play_data()
        
        # Check that data was added
        assert new_transitions > 0
        assert len(training_loop.replay_buffer) > initial_buffer_size
        assert training_loop.episode == 2  # 2 games played


def test_training_loop_training_step():
    """Test training step functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            batch_size=4,
            min_replay_size=5
        )
        
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        # Should not train with insufficient data
        metrics = training_loop._training_step()
        assert metrics is None
        
        # Add some data to buffer
        training_loop._generate_self_play_data()
        
        # Should be able to train if buffer has enough data
        if len(training_loop.replay_buffer) >= config.min_replay_size:
            metrics = training_loop._training_step()
            assert metrics is not None
            assert 'loss' in metrics


def test_training_loop_evaluation():
    """Test agent evaluation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            eval_games=4  # Small number for fast testing
        )
        
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        # Run evaluation
        eval_metrics = training_loop._evaluate_agent()
        
        # Check evaluation results
        assert 'random_win_rate' in eval_metrics
        assert 'heuristic_win_rate' in eval_metrics
        assert 'random_draw_rate' in eval_metrics
        assert 'heuristic_draw_rate' in eval_metrics
        
        # Win rates should be between 0 and 1
        assert 0 <= eval_metrics['random_win_rate'] <= 1
        assert 0 <= eval_metrics['heuristic_win_rate'] <= 1


def test_training_loop_checkpointing():
    """Test model checkpointing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(total_steps=10)
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        # Set some training state
        training_loop.step = 5
        training_loop.episode = 3
        
        # Save checkpoint
        training_loop._save_checkpoint()
        
        # Check that files were created
        checkpoint_files = list(training_loop.models_dir.glob("*step_5*"))
        assert len(checkpoint_files) >= 2  # trainer and agent checkpoints
        
        # Test saving as best
        training_loop._save_checkpoint(is_best=True)
        
        best_files = list(training_loop.models_dir.glob("best_*"))
        assert len(best_files) >= 2  # best trainer and agent


def test_training_loop_logging():
    """Test metrics logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig()
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        training_loop.step = 10
        training_loop.episode = 5
        
        # Log some metrics
        training_metrics = {'loss': 0.5, 'q_mean': 1.0}
        eval_metrics = {'random_win_rate': 0.8}
        
        training_loop._log_metrics(training_metrics, eval_metrics)
        
        # Check that metrics were stored
        assert len(training_loop.training_metrics) == 1
        assert training_loop.training_metrics[0]['step'] == 10
        assert training_loop.training_metrics[0]['loss'] == 0.5
        
        # Check that log file was created
        log_file = training_loop.logs_dir / "training_metrics.jsonl"
        assert log_file.exists()
        
        with open(log_file) as f:
            logged_data = json.loads(f.readline())
        assert logged_data['step'] == 10
        assert logged_data['loss'] == 0.5


def test_training_loop_short_training_run():
    """Test a very short training run end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            total_steps=5,
            batch_size=2,
            games_per_update=1,
            min_replay_size=3,
            train_every=2,
            eval_every=10,  # No evaluation in short run
            save_every=10,  # No saving in short run
            log_every=2
        )
        
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        # Run training
        history = training_loop.train()
        
        # Check that training completed
        assert training_loop.step == 5
        assert training_loop.episode > 0
        
        # Check history
        assert 'training_metrics' in history
        assert 'config' in history
        assert 'run_dir' in history
        
        # Should have some logged metrics
        assert len(training_loop.training_metrics) >= 1


def test_training_loop_with_custom_run_name():
    """Test training loop with custom run name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(total_steps=1)
        
        training_loop = TrainingLoop(
            config=config,
            data_dir=tmpdir,
            run_name="custom_test_run"
        )
        
        assert "custom_test_run" in str(training_loop.run_dir)


def test_training_loop_default_data_dir():
    """Test training loop with default data directory."""
    config = TrainingConfig(total_steps=1)
    
    training_loop = TrainingLoop(config=config)
    
    # Should use default data directory
    assert training_loop.run_dir.exists()
    assert "data" in str(training_loop.run_dir)


def test_training_config_defaults():
    """Test that training config has reasonable defaults."""
    config = TrainingConfig()
    
    # Check key defaults
    assert config.total_steps > 0
    assert config.batch_size > 0
    assert config.learning_rate > 0
    assert 0 < config.gamma <= 1
    assert config.tau > 0
    assert config.replay_capacity > config.batch_size
    assert config.min_replay_size > 0
    assert config.device in ['cpu', 'cuda']


def test_training_loop_component_setup():
    """Test that all training components are properly set up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig()
        training_loop = TrainingLoop(config=config, data_dir=tmpdir)
        
        # Check agent setup
        assert training_loop.main_agent.board_size == 15
        assert training_loop.main_agent.epsilon == config.self_play_epsilon
        
        # Check networks
        assert training_loop.online_network is not None
        assert training_loop.target_network is not None
        assert training_loop.online_network is training_loop.main_agent.model
        
        # Check trainer
        assert training_loop.trainer.online_network is training_loop.online_network
        assert training_loop.trainer.target_network is training_loop.target_network
        assert training_loop.trainer.gamma == config.gamma
        
        # Check replay buffer
        assert training_loop.replay_buffer.capacity == config.replay_capacity
        assert training_loop.replay_buffer.batch_size == config.batch_size
        
        # Check evaluation agents
        assert 'random' in training_loop.eval_agents
        assert 'heuristic' in training_loop.eval_agents