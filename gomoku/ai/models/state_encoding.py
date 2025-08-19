"""
State encoding utilities for DQN Gomoku agent.

Converts game state into 4-channel tensor format for neural network input.
"""
import numpy as np
import torch


def encode_game_state(game, last_move=None):
    """
    Encode game state into 4-channel tensor format for DQN.
    
    Args:
        game: Game instance with current state
        last_move: Optional tuple (row, col) of last move made
        
    Returns:
        torch.Tensor: Shape (4, 15, 15) with channels:
            - Channel 0: Current player stones (1 where current player has stones, 0 elsewhere)
            - Channel 1: Opponent stones (1 where opponent has stones, 0 elsewhere)  
            - Channel 2: Turn plane (1 for black-to-play, 0 for white-to-play, constant across board)
            - Channel 3: Last-move plane (1 at last move position, 0 elsewhere)
    """
    board_size = 15
    channels = 4
    
    # Initialize 4-channel state tensor
    state = np.zeros((channels, board_size, board_size), dtype=np.float32)
    
    # Get current player and opponent values
    current_player = game.current_player  # 1 for black, -1 for white
    opponent_player = -current_player
    
    # Channel 0: Current player stones
    state[0] = (game.board.state == current_player).astype(np.float32)
    
    # Channel 1: Opponent stones  
    state[1] = (game.board.state == opponent_player).astype(np.float32)
    
    # Channel 2: Turn plane (1 if black to play, 0 if white to play)
    turn_value = 1.0 if current_player == 1 else 0.0
    state[2].fill(turn_value)
    
    # Channel 3: Last-move plane
    if last_move is not None:
        row, col = last_move
        if 0 <= row < board_size and 0 <= col < board_size:
            state[3, row, col] = 1.0
    
    return torch.tensor(state, dtype=torch.float32)


def decode_action_index(action_idx, board_size=15):
    """
    Convert action index to (row, col) coordinates.
    
    Args:
        action_idx: Integer index (0-224 for 15x15 board)
        board_size: Board size (default 15)
        
    Returns:
        tuple: (row, col) coordinates
    """
    row = action_idx // board_size
    col = action_idx % board_size
    return (row, col)


def encode_action_coordinates(row, col, board_size=15):
    """
    Convert (row, col) coordinates to action index.
    
    Args:
        row: Row coordinate (0-14)
        col: Column coordinate (0-14)  
        board_size: Board size (default 15)
        
    Returns:
        int: Action index (0-224 for 15x15 board)
    """
    return row * board_size + col


def get_legal_moves_mask(game, board_size=15):
    """
    Create boolean mask for legal moves.
    
    Args:
        game: Game instance
        board_size: Board size (default 15)
        
    Returns:
        torch.Tensor: Shape (225,) boolean mask where True = legal move
    """
    legal_moves = game.board.get_legal_moves()
    mask = torch.zeros(board_size * board_size, dtype=torch.bool)
    
    for row, col in legal_moves:
        action_idx = encode_action_coordinates(row, col, board_size)
        mask[action_idx] = True
        
    return mask