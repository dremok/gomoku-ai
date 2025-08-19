#!/usr/bin/env python3
"""
Quick demonstration of local data directory usage.

This script shows how the training data now gets saved to a local data/ directory
instead of temporary system directories.
"""
import sys
import os

# Add the parent directory to Python path so we can import gomoku
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gomoku.ai.training.data_utils import get_default_data_dir, create_training_dataset


def main():
    """Demonstrate local data directory usage."""
    print("🗂️  Local Data Directory Demo")
    print("=" * 40)
    
    # Show where data will be stored
    data_dir = get_default_data_dir()
    print(f"Default data directory: {data_dir}")
    print(f"Data directory exists: {data_dir.exists()}")
    
    # Create a small training dataset using defaults
    print("\nCreating small training dataset...")
    print("(Using default agent and data directory)")
    
    buffer, batch_generator = create_training_dataset(
        num_games=3,  # Very small for quick demo
        batch_size=2
    )
    
    print(f"\n✅ Dataset created!")
    print(f"Buffer size: {len(buffer)} transitions")
    
    # Show what's in the data directory
    print(f"\nContents of {data_dir}:")
    if data_dir.exists():
        for item in data_dir.rglob("*"):
            if item.is_file():
                print(f"  📄 {item.relative_to(data_dir)}")
            elif item.is_dir() and item != data_dir:
                print(f"  📁 {item.relative_to(data_dir)}/")
    
    print(f"\n💡 All training data is now stored locally in the project!")
    print(f"   This directory is git-ignored so it won't be committed.")
    print(f"   You can safely delete {data_dir} to clean up training data.")


if __name__ == "__main__":
    main()