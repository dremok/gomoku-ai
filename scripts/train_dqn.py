#!/usr/bin/env python3
"""
DQN training script for Gomoku AI.

This script trains a Deep Q-Network agent using self-play and experience replay.
"""
import sys
import os
import argparse
import json
from pathlib import Path

# Add the parent directory to Python path so we can import gomoku
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gomoku.ai.training.training_loop import TrainingConfig, TrainingLoop
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN agent for Gomoku')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=10000, 
                       help='Total training steps (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    
    # Experience replay
    parser.add_argument('--replay-capacity', type=int, default=50000,
                       help='Replay buffer capacity (default: 50000)')
    parser.add_argument('--min-replay', type=int, default=1000,
                       help='Minimum replay size before training (default: 1000)')
    
    # Self-play
    parser.add_argument('--games-per-update', type=int, default=10,
                       help='Games per training update (default: 10)')
    parser.add_argument('--epsilon', type=float, default=0.3,
                       help='Initial exploration epsilon (default: 0.3)')
    parser.add_argument('--epsilon-min', type=float, default=0.05,
                       help='Minimum exploration epsilon (default: 0.05)')
    parser.add_argument('--epsilon-decay', type=int, default=None,
                       help='Epsilon decay steps (default: 50% of total steps)')
    
    # Training schedule
    parser.add_argument('--train-every', type=int, default=4,
                       help='Train every N steps (default: 4)')
    parser.add_argument('--eval-every', type=int, default=1000,
                       help='Evaluate every N steps (default: 1000)')
    parser.add_argument('--save-every', type=int, default=5000,
                       help='Save checkpoint every N steps (default: 5000)')
    
    # Evaluation
    parser.add_argument('--eval-games', type=int, default=100,
                       help='Games per evaluation (default: 100)')
    
    # Logging
    parser.add_argument('--log-every', type=int, default=100,
                       help='Log metrics every N steps (default: 100)')
    
    # Output
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory (default: auto)')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Run name (default: auto-generated)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Training device (default: auto - detects MPS > CUDA > CPU)')
    
    # Preset configurations
    parser.add_argument('--preset', type=str, choices=['demo', 'quick', 'fast', 'turbo', 'strategic', 'standard', 'long'],
                       help='Use preset configuration')
    
    # Training visibility
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed training progress')
    
    # Reward structure
    parser.add_argument('--tactical-rewards', action='store_true',
                       help='Enable tactical rewards for threat creation/blocking')
    
    # Config file
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')
    
    return parser.parse_args()


def get_preset_config(preset: str) -> dict:
    """Get preset training configuration."""
    presets = {
        'demo': {
            'total_steps': 500,
            'batch_size': 16,
            'replay_capacity': 2000,
            'min_replay_size': 50,
            'games_per_update': 3,
            'eval_every': 100,
            'save_every': 250,
            'eval_games': 50,
            'log_every': 25,
            'train_every': 2,
            'epsilon_decay_steps': 250
        },
        'quick': {
            'total_steps': 2000,
            'batch_size': 16,
            'replay_capacity': 10000,
            'min_replay_size': 200,
            'games_per_update': 5,
            'eval_every': 200,
            'save_every': 500,
            'eval_games': 50,
            'log_every': 50,
            'epsilon_decay_steps': 1000
        },
        'fast': {
            'total_steps': 3000,
            'batch_size': 64,  # Larger batches for efficiency
            'replay_capacity': 15000,
            'min_replay_size': 200,  # Start training sooner
            'games_per_update': 4,  # Fewer games per update
            'eval_every': 1000,  # Less frequent evaluation
            'save_every': 1500,
            'eval_games': 30,  # Fewer evaluation games
            'log_every': 100,
            'train_every': 2,  # More frequent training updates
            'epsilon_decay_steps': 2000,
            'learning_rate': 2e-3,  # Higher learning rate for faster convergence
            'target_update_every': 8,  # Less frequent target updates
            'tactical_rewards': False,  # Disable complex reward calculation
            'opponent_weights': [0.8, 0.15, 0.05]  # More self-play, minimal heuristic overhead
        },
        'turbo': {
            'total_steps': 2000,
            'batch_size': 128,  # Very large batches
            'replay_capacity': 10000,
            'min_replay_size': 128,  # Minimal warmup
            'games_per_update': 2,  # Minimal games per update
            'eval_every': 500,  # Very infrequent evaluation
            'save_every': 1000,
            'eval_games': 20,  # Minimal evaluation games
            'log_every': 50,
            'train_every': 1,  # Train every step
            'epsilon_decay_steps': 1200,
            'learning_rate': 5e-3,  # Very high learning rate
            'target_update_every': 10,  # Infrequent target updates
            'tactical_rewards': False,
            'opponent_weights': [0.9, 0.1, 0.0]  # Pure self-play, no heuristic
        },
        'strategic': {
            'total_steps': 5000,
            'batch_size': 32,
            'replay_capacity': 25000,
            'min_replay_size': 500,
            'games_per_update': 8,
            'eval_every': 250,
            'save_every': 1000,
            'eval_games': 100,
            'log_every': 50,
            'train_every': 3,
            'epsilon_decay_steps': 3000,
            'learning_rate': 5e-4,  # Slightly lower for more stable learning
            'target_update_every': 2,  # More frequent target updates
            'tactical_rewards': False,  # Disabled temporarily - needs tensor state fix
            'opponent_weights': [0.6, 0.1, 0.3]  # 60% self-play, 10% random, 30% heuristic
        },
        'standard': {
            'total_steps': 10000,
            'batch_size': 32,
            'replay_capacity': 50000,
            'min_replay_size': 1000,
            'games_per_update': 10,
            'eval_every': 500,
            'save_every': 2000,
            'eval_games': 100,
            'log_every': 100,
            'epsilon_decay_steps': 5000
        },
        'long': {
            'total_steps': 50000,
            'batch_size': 64,
            'replay_capacity': 200000,
            'min_replay_size': 5000,
            'games_per_update': 20,
            'eval_every': 1000,
            'save_every': 5000,
            'eval_games': 200,
            'log_every': 200,
            'epsilon_decay_steps': 25000
        }
    }
    return presets.get(preset, {})


def detect_best_device() -> str:
    """Detect the best available device for training."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_config_from_args(args) -> TrainingConfig:
    """Create training config from command line arguments."""
    config_dict = {}
    
    # Load base config
    if args.config:
        config_dict.update(load_config_from_file(args.config))
    elif args.preset:
        config_dict.update(get_preset_config(args.preset))
    
    # Override with command line arguments
    if args.steps != 10000:  # Non-default value
        config_dict['total_steps'] = args.steps
    if args.batch_size != 32:
        config_dict['batch_size'] = args.batch_size
    if args.lr != 1e-3:
        config_dict['learning_rate'] = args.lr
    if args.gamma != 0.99:
        config_dict['gamma'] = args.gamma
    
    if args.replay_capacity != 50000:
        config_dict['replay_capacity'] = args.replay_capacity
    if args.min_replay != 1000:
        config_dict['min_replay_size'] = args.min_replay
    
    if args.games_per_update != 10:
        config_dict['games_per_update'] = args.games_per_update
    if args.epsilon != 0.3:
        config_dict['self_play_epsilon'] = args.epsilon
    if args.epsilon_min != 0.05:
        config_dict['epsilon_min'] = args.epsilon_min
    
    if args.epsilon_decay:
        config_dict['epsilon_decay_steps'] = args.epsilon_decay
    elif 'epsilon_decay_steps' not in config_dict:
        # Default to 50% of total steps
        config_dict['epsilon_decay_steps'] = config_dict.get('total_steps', args.steps) // 2
    
    if args.train_every != 4:
        config_dict['train_every'] = args.train_every
    if args.eval_every != 1000:
        config_dict['eval_every'] = args.eval_every
    if args.save_every != 5000:
        config_dict['save_every'] = args.save_every
    
    if args.eval_games != 100:
        config_dict['eval_games'] = args.eval_games
    
    if args.log_every != 100:
        config_dict['log_every'] = args.log_every
    
    # Handle device selection
    if args.device == 'auto':
        selected_device = detect_best_device()
        config_dict['device'] = selected_device
    elif args.device != 'cpu':
        config_dict['device'] = args.device
    
    return TrainingConfig(**config_dict)


def print_config_summary(config: TrainingConfig, preset_name: str = None):
    """Print a summary of the training configuration."""
    title = "🎮 DQN Training Configuration"
    if preset_name:
        title += f" ({preset_name.upper()} preset)"
    print(title)
    print("=" * 60)
    print(f"Training Steps:      {config.total_steps:,}")
    print(f"Batch Size:          {config.batch_size}")
    print(f"Learning Rate:       {config.learning_rate}")
    print(f"Discount Factor:     {config.gamma}")
    print(f"Replay Capacity:     {config.replay_capacity:,}")
    print(f"Min Replay Size:     {config.min_replay_size:,}")
    print(f"Games per Update:    {config.games_per_update}")
    print(f"Initial Epsilon:     {config.self_play_epsilon}")
    print(f"Final Epsilon:       {config.epsilon_min}")
    print(f"Epsilon Decay:       {config.epsilon_decay_steps:,} steps")
    print(f"Evaluation Games:    {config.eval_games}")
    print(f"Device:              {config.device}")
    print("=" * 60)


def estimate_training_time(config: TrainingConfig):
    """Estimate training time based on configuration."""
    # Rough estimates based on typical performance
    seconds_per_step = 0.1  # Very rough estimate
    total_seconds = config.total_steps * seconds_per_step
    
    hours = total_seconds / 3600
    if hours < 1:
        time_str = f"{total_seconds/60:.1f} minutes"
    else:
        time_str = f"{hours:.1f} hours"
    
    print(f"💡 Estimated training time: ~{time_str}")
    print(f"   (This is a rough estimate - actual time may vary)")
    print()


def main():
    """Main training function."""
    args = parse_args()
    
    # Use demo preset as default if no arguments provided
    if not any(vars(args).values()) or (len(sys.argv) == 1):
        print("🎯 No arguments provided - using 'demo' preset for quick training!")
        print("   Speed presets: --preset turbo/fast (optimized for speed)")
        print("   Quality presets: --preset strategic/standard/long")
        print("   Or run with --help for all options\n")
        args.preset = 'demo'
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Show device info
    if args.device == 'auto':
        print(f"🔍 Auto-detected device: {config.device}")
        if config.device == 'mps':
            print("   Using Apple Silicon GPU (Metal Performance Shaders)")
        elif config.device == 'cuda':
            print("   Using NVIDIA GPU")
        else:
            print("   Using CPU (no GPU available)")
    else:
        print(f"🔧 Using specified device: {config.device}")
    
    print()
    
    # Print summary
    print_config_summary(config, args.preset)
    estimate_training_time(config)
    
    # Create and run training loop
    try:
        training_loop = TrainingLoop(
            config=config,
            data_dir=args.data_dir,
            run_name=args.run_name
        )
        
        print(f"📁 Training data will be saved to: {training_loop.run_dir}")
        print()
        
        # Run training
        history = training_loop.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Run data saved to: {history['run_dir']}")
        print(f"🏆 Best models saved in: {training_loop.models_dir}")
        
        # Print final performance summary
        if history['training_metrics']:
            final_metrics = history['training_metrics'][-1]
            print(f"\n📈 Final Performance:")
            if 'random_win_rate' in final_metrics:
                print(f"   vs Random:    {final_metrics['random_win_rate']:.1%}")
            if 'heuristic_win_rate' in final_metrics:
                print(f"   vs Heuristic: {final_metrics['heuristic_win_rate']:.1%}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Partial results may be saved in the training directory")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()