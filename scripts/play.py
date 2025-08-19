#!/usr/bin/env python3
"""
CLI interface for playing Gomoku against humans or AI agents.
"""
import sys
import os
import time

# Add the parent directory to Python path so we can import gomoku
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gomoku.core.game import Game
from gomoku.ai.agents.random_agent import RandomAgent
from gomoku.ai.agents.heuristic_agent import HeuristicAgent


def display_board(game):
    """Display the current board state in ASCII format."""
    print("\n   ", end="")
    # Column headers
    for col in range(15):
        print(f"{col:2d}", end=" ")
    print()
    
    print("   " + "---" * 15)
    
    # Board rows
    for row in range(15):
        print(f"{row:2d}|", end="")
        for col in range(15):
            cell = game.board.state[row, col]
            if cell == 1:
                print(" X", end=" ")  # Black
            elif cell == -1:
                print(" O", end=" ")  # White  
            else:
                print(" .", end=" ")  # Empty
        print(f"|{row:2d}")
    
    print("   " + "---" * 15)
    print("   ", end="")
    for col in range(15):
        print(f"{col:2d}", end=" ")
    print()


def get_player_name(player):
    """Get display name for player."""
    return "Black (X)" if player == 1 else "White (O)"


def parse_move(move_input):
    """
    Parse move input from user.
    
    Args:
        move_input (str): User input like "7 7" or "7,7"
        
    Returns:
        tuple: (row, col) or None if invalid
    """
    try:
        # Handle both space and comma separated input
        if ',' in move_input:
            parts = move_input.split(',')
        else:
            parts = move_input.split()
            
        if len(parts) != 2:
            return None
            
        row = int(parts[0].strip())
        col = int(parts[1].strip())
        
        # Validate range
        if 0 <= row < 15 and 0 <= col < 15:
            return (row, col)
        else:
            return None
            
    except ValueError:
        return None


def main():
    """Main game loop."""
    print("=" * 50)
    print("         GOMOKU (Standard 5-only Rules)")
    print("=" * 50)
    print("Rules: Get exactly 5 stones in a row to win.")
    print("Overlines (6+ stones) do NOT count as wins!")
    print("Black (X) goes first. Enter moves as: row col")
    print("Example: '7 7' places stone at center")
    print("=" * 50)
    
    game = Game()
    
    while game.game_state == 'ongoing':
        display_board(game)
        print(f"\nCurrent player: {get_player_name(game.current_player)}")
        print(f"Legal moves remaining: {len(game.board.get_legal_moves())}")
        
        # Get move from user
        while True:
            try:
                move_input = input("Enter your move (row col) or 'quit' to exit: ").strip()
                
                if move_input.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for playing!")
                    return
                    
                move = parse_move(move_input)
                if move is None:
                    print("Invalid input! Please enter: row col (e.g., '7 7')")
                    continue
                    
                row, col = move
                if game.make_move(row, col):
                    break
                else:
                    print(f"Invalid move! Position ({row}, {col}) is not available.")
                    print("Make sure the position is empty and within bounds (0-14).")
                    
            except KeyboardInterrupt:
                print("\nThanks for playing!")
                return
            except EOFError:
                print("\nThanks for playing!")
                return
    
    # Game ended - show final state
    display_board(game)
    print("\n" + "=" * 50)
    
    if game.game_state == 'win':
        winner_name = get_player_name(game.winner)
        print(f"🎉 GAME OVER - {winner_name} wins!")
        print("Congratulations!")
    elif game.game_state == 'draw':
        print("🤝 GAME OVER - It's a draw!")
        print("The board is full with no winner.")
    
    print("=" * 50)


if __name__ == "__main__":
    main()