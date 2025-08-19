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
from gomoku.ai.agents.dqn_agent import DQNAgent


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


def select_game_mode():
    """
    Let user select game mode.
    
    Returns:
        tuple: (mode, player1, player2) where mode is string and players are agent instances or None for human
    """
    print("\nSelect Game Mode:")
    print("1. Player vs Player (PvP)")
    print("2. Player vs AI - Random (Easy)")
    print("3. Player vs AI - Heuristic (Hard)")
    print("4. Player vs AI - DQN (Neural Network - Untrained)")
    print("5. AI vs AI - Random vs Heuristic")
    print("6. AI vs AI - Heuristic vs DQN")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                return ('pvp', None, None)
            elif choice == '2':
                return ('pvai_random', None, RandomAgent(seed=42))
            elif choice == '3':
                return ('pvai_heuristic', None, HeuristicAgent(seed=42))
            elif choice == '4':
                return ('pvai_dqn', None, DQNAgent(epsilon=0.1, seed=42))
            elif choice == '5':
                return ('aivai_random_heuristic', RandomAgent(seed=123), HeuristicAgent(seed=456))
            elif choice == '6':
                return ('aivai_heuristic_dqn', HeuristicAgent(seed=123), DQNAgent(epsilon=0.1, seed=456))
            else:
                print("Invalid choice! Please enter 1, 2, 3, 4, 5, or 6.")
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)


def get_human_move(game, player):
    """
    Get move input from human player.
    
    Returns:
        tuple: (row, col) or None if quit
    """
    while True:
        try:
            player_name = get_player_name(player)
            move_input = input(f"{player_name}, enter your move (row col) or 'quit': ").strip()
            
            if move_input.lower() in ['quit', 'exit', 'q']:
                return None
                
            move = parse_move(move_input)
            if move is None:
                print("Invalid input! Please enter: row col (e.g., '7 7')")
                continue
                
            row, col = move
            if game.board.state[row, col] == 0:  # Position is empty
                return (row, col)
            else:
                print(f"Position ({row}, {col}) is already occupied!")
                
        except (KeyboardInterrupt, EOFError):
            return None


def get_ai_move(agent, game, player):
    """
    Get move from AI agent.
    
    Returns:
        tuple: (row, col)
    """
    print(f"{get_player_name(player)} (AI) is thinking...")
    
    # Add small delay to make it feel more natural
    time.sleep(0.5)
    
    move = agent.select_action(game)
    if move:
        print(f"{get_player_name(player)} (AI) plays: {move[0]} {move[1]}")
    return move


def main():
    """Main game loop."""
    print("=" * 60)
    print("           GOMOKU (Standard 5-only Rules)")
    print("=" * 60)
    print("Rules: Get exactly 5 stones in a row to win.")
    print("Overlines (6+ stones) do NOT count as wins!")
    print("Black (X) goes first. Enter moves as: row col")
    print("Example: '7 7' places stone at center")
    print("=" * 60)
    
    # Select game mode
    mode, player1_agent, player2_agent = select_game_mode()
    
    # Display mode info
    mode_descriptions = {
        'pvp': "Player vs Player",
        'pvai_random': "Player vs AI (Random - Easy)",
        'pvai_heuristic': "Player vs AI (Heuristic - Hard)",
        'pvai_dqn': "Player vs AI (DQN - Neural Network)",
        'aivai_random_heuristic': "AI vs AI (Random vs Heuristic)",
        'aivai_heuristic_dqn': "AI vs AI (Heuristic vs DQN)"
    }
    
    print(f"\n🎮 Starting: {mode_descriptions[mode]}")
    print("=" * 60)
    
    game = Game()
    move_count = 0
    
    try:
        while game.game_state == 'ongoing':
            display_board(game)
            print(f"\nMove #{move_count + 1}")
            print(f"Current player: {get_player_name(game.current_player)}")
            print(f"Legal moves remaining: {len(game.board.get_legal_moves())}")
            
            # Determine current player's agent
            if game.current_player == 1:  # Black player
                current_agent = player1_agent
            else:  # White player  
                current_agent = player2_agent
                
            # Get move based on whether current player is human or AI
            if current_agent is None:  # Human player
                move = get_human_move(game, game.current_player)
                if move is None:  # User quit
                    print("\nThanks for playing!")
                    return
            else:  # AI player
                move = get_ai_move(current_agent, game, game.current_player)
                if move is None:  # Should not happen but handle gracefully
                    print("AI could not find a move! Game ending.")
                    break
                    
            # Make the move
            if game.make_move(*move):
                move_count += 1
                # Add small pause after AI moves for better UX
                if current_agent is not None and mode != 'aivai':
                    time.sleep(1)
            else:
                print(f"ERROR: Invalid move {move}!")
                break
    
    except (KeyboardInterrupt, EOFError):
        print("\nThanks for playing!")
        return
    
    # Game ended - show final state
    display_board(game)
    print("\n" + "=" * 60)
    
    if game.game_state == 'win':
        winner_name = get_player_name(game.winner)
        if mode == 'pvp':
            print(f"🎉 GAME OVER - {winner_name} wins!")
        elif mode.startswith('pvai'):
            if (game.winner == 1 and player1_agent is None) or (game.winner == -1 and player2_agent is None):
                print(f"🎉 CONGRATULATIONS! You ({winner_name}) beat the AI!")
            else:
                print(f"💻 AI ({winner_name}) wins! Better luck next time!")
        elif mode.startswith('aivai'):
            if mode == 'aivai_random_heuristic':
                ai_name = "Random AI" if game.winner == 1 else "Heuristic AI"
            elif mode == 'aivai_heuristic_dqn':
                ai_name = "Heuristic AI" if game.winner == 1 else "DQN AI"
            else:
                ai_name = "AI"
            print(f"🤖 {ai_name} ({winner_name}) wins!")
            
        print(f"Game completed in {move_count} moves.")
        
    elif game.game_state == 'draw':
        print("🤝 GAME OVER - It's a draw!")
        print("The board is full with no winner.")
        print(f"Game completed in {move_count} moves.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()