#!/usr/bin/env python3
"""
Demonstration script for self-play and replay buffer systems.

This script shows how to:
1. Generate training data through self-play
2. Load and analyze the data
3. Prepare batches for training
"""
import sys
import os
import tempfile

# Add the parent directory to Python path so we can import gomoku
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gomoku.ai.agents.dqn_agent import DQNAgent
from gomoku.ai.training.self_play import SelfPlayManager, SelfPlayGame
from gomoku.ai.training.replay_buffer import ReplayBuffer
from gomoku.ai.training.data_utils import (
    TrainingDataLoader, 
    TrainingDataGenerator, 
    TrainingBatchGenerator,
    create_training_dataset
)


def demo_simple_self_play():
    """Demonstrate a simple self-play game."""
    print("=== Demo: Simple Self-Play Game ===")
    
    # Create agents
    dqn_agent = DQNAgent(seed=42, epsilon=0.3)
    
    # Play a single game
    game = SelfPlayGame(
        player1_agent=dqn_agent,
        player2_agent=dqn_agent,  # Self-play
        collect_data=True,
        verbose=True
    )
    
    result = game.play_game()
    transitions = game.get_transitions()
    
    print(f"\nGame result: {result['outcome']}")
    print(f"Winner: {result['winner']}")
    print(f"Moves: {result['moves']}")
    print(f"Transitions collected: {len(transitions)}")
    
    # Show some transition details
    if transitions:
        first_transition = transitions[0]
        print(f"\nFirst transition:")
        print(f"  State shape: {first_transition.state.shape}")
        print(f"  Action: {first_transition.action}")
        print(f"  Reward: {first_transition.reward}")
        print(f"  Done: {first_transition.done}")


def demo_self_play_manager():
    """Demonstrate the self-play manager."""
    print("\n=== Demo: Self-Play Manager ===")
    
    # Create main agent
    main_agent = DQNAgent(seed=42, epsilon=0.2)
    
    # Create manager
    manager = SelfPlayManager(
        main_agent=main_agent,
        games_per_iteration=5,
        verbose=True
    )
    
    # Play multiple games
    print("Playing 5 self-play games...")
    results, transitions = manager.play_games(
        num_games=5,
        collect_data=True
    )
    
    print(f"\nCollected {len(transitions)} transitions from {len(results)} games")
    
    # Analyze results
    outcomes = [r['outcome'] for r in results]
    print(f"Outcomes: {outcomes}")
    
    # Show win rate
    wins = sum(1 for r in results if r['outcome'] == 'win' and 
              ((r['winner'] == 1 and r['player1_agent'] == 'DQNAgent') or
               (r['winner'] == -1 and r['player2_agent'] == 'DQNAgent')))
    win_rate = wins / len(results) if results else 0
    print(f"DQN agent win rate: {win_rate:.2%}")


def demo_replay_buffer():
    """Demonstrate replay buffer functionality."""
    print("\n=== Demo: Replay Buffer ===")
    
    # Generate some training data first
    main_agent = DQNAgent(seed=42, epsilon=0.5)
    manager = SelfPlayManager(main_agent=main_agent, verbose=False)
    
    print("Generating training data...")
    results, transitions = manager.play_games(num_games=3, collect_data=True)
    
    # Create replay buffer
    buffer = ReplayBuffer(capacity=1000, batch_size=4)
    
    # Add transitions
    buffer.add_trajectory(transitions)
    
    print(f"Buffer size: {len(buffer)} transitions")
    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Can sample: {buffer.can_sample()}")
    
    # Sample a batch
    if buffer.can_sample():
        states, actions, rewards, next_states, dones = buffer.sample()
        
        print(f"\nSampled batch:")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Next states shape: {next_states.shape}")
        print(f"  Dones shape: {dones.shape}")
        
        print(f"  Sample actions: {actions.tolist()}")
        print(f"  Sample rewards: {rewards.tolist()}")
        print(f"  Sample dones: {dones.tolist()}")


def demo_training_data_generation():
    """Demonstrate complete training data generation."""
    print("\n=== Demo: Training Data Generation ===")
    
    # Use local data directory instead of temp
    from gomoku.ai.training.data_utils import get_default_data_dir
    data_dir = get_default_data_dir() / "demo"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Using local data directory: {data_dir}")
    
    # Create agent
    agent = DQNAgent(seed=42, epsilon=0.4)
    
    # Generate training dataset
    print("Creating training dataset...")
    buffer, batch_generator = create_training_dataset(
        data_dir=str(data_dir),
        main_agent=agent,
        num_games=10,  # Small number for demo
        batch_size=4,
        device='cpu'
    )
        
    print(f"Dataset created!")
    print(f"Buffer size: {len(buffer)} transitions")
    
    # Generate some training batches
    print("\nGenerating training batches...")
    batch_count = 0
    for batch in batch_generator.batch_iterator(num_batches=3):
        states, actions, rewards, next_states, dones = batch
        batch_count += 1
        print(f"Batch {batch_count}: {states.shape[0]} samples")
    
    # Show data statistics
    stats = batch_generator.get_data_statistics()
    print(f"\nBatch generator statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Load and analyze saved data
    loader = TrainingDataLoader(str(data_dir))
    data_stats = loader.get_data_statistics()
    print(f"\nData directory statistics:")
    print(f"  Total transitions: {data_stats['total_transitions']}")
    print(f"  Total games: {data_stats['total_games']}")
    print(f"  Replay buffer files: {data_stats['file_counts']['replay_buffers']}")
    
    print(f"\nTraining data saved to: {data_dir}")
    print("You can find replay buffers, game results, and session stats in the data directory.")


def main():
    """Run all demonstrations."""
    print("🎮 Gomoku Self-Play & Replay System Demo")
    print("=" * 50)
    
    try:
        demo_simple_self_play()
        demo_self_play_manager()
        demo_replay_buffer()
        demo_training_data_generation()
        
        print("\n✅ All demos completed successfully!")
        print("\nThe self-play and replay system is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()