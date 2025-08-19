"""
Action selection utilities for DQN agent.

Implements ε-greedy action selection with legal move masking.
"""
import torch
import random
import numpy as np
from typing import Union, Tuple, Optional


def epsilon_greedy_action(q_values: torch.Tensor, 
                         legal_moves_mask: torch.Tensor,
                         epsilon: float,
                         rng: Optional[random.Random] = None) -> int:
    """
    Select action using ε-greedy policy with legal move masking.
    
    Args:
        q_values: Q-values tensor of shape (num_actions,) - typically 225 for 15x15 board
        legal_moves_mask: Boolean mask of shape (num_actions,) where True = legal move
        epsilon: Exploration rate (0.0 = fully greedy, 1.0 = fully random)
        rng: Optional random number generator for reproducible behavior
        
    Returns:
        int: Selected action index
        
    Raises:
        ValueError: If no legal moves are available
    """
    if rng is None:
        rng = random.Random()
    
    # Ensure we have legal moves
    legal_indices = torch.nonzero(legal_moves_mask, as_tuple=True)[0]
    if len(legal_indices) == 0:
        raise ValueError("No legal moves available")
    
    # ε-greedy decision
    if rng.random() < epsilon:
        # Exploration: random legal move
        random_idx = rng.randint(0, len(legal_indices) - 1)
        action_idx = legal_indices[random_idx].item()
    else:
        # Exploitation: best legal move according to Q-values
        # Mask out illegal moves by setting them to very negative values
        masked_q_values = q_values.clone()
        masked_q_values[~legal_moves_mask] = float('-inf')
        
        # Select action with highest Q-value
        action_idx = torch.argmax(masked_q_values).item()
    
    return action_idx


def epsilon_greedy_action_batch(q_values: torch.Tensor,
                               legal_moves_masks: torch.Tensor,
                               epsilon: float,
                               rng: Optional[random.Random] = None) -> torch.Tensor:
    """
    Select actions using ε-greedy policy for a batch of states.
    
    Args:
        q_values: Q-values tensor of shape (batch_size, num_actions)
        legal_moves_masks: Boolean mask of shape (batch_size, num_actions)
        epsilon: Exploration rate (0.0 = fully greedy, 1.0 = fully random)
        rng: Optional random number generator for reproducible behavior
        
    Returns:
        torch.Tensor: Selected actions of shape (batch_size,) with dtype long
        
    Raises:
        ValueError: If any sample has no legal moves
    """
    if rng is None:
        rng = random.Random()
    
    batch_size, num_actions = q_values.shape
    selected_actions = torch.zeros(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        # Check for legal moves
        legal_indices = torch.nonzero(legal_moves_masks[i], as_tuple=True)[0]
        if len(legal_indices) == 0:
            raise ValueError(f"No legal moves available for sample {i}")
        
        # ε-greedy decision for this sample
        if rng.random() < epsilon:
            # Exploration: random legal move
            random_idx = rng.randint(0, len(legal_indices) - 1)
            selected_actions[i] = legal_indices[random_idx]
        else:
            # Exploitation: best legal move according to Q-values
            masked_q_values = q_values[i].clone()
            masked_q_values[~legal_moves_masks[i]] = float('-inf')
            selected_actions[i] = torch.argmax(masked_q_values)
    
    return selected_actions


def greedy_action(q_values: torch.Tensor, 
                 legal_moves_mask: torch.Tensor) -> int:
    """
    Select the best legal action (fully greedy, no exploration).
    
    Args:
        q_values: Q-values tensor of shape (num_actions,)
        legal_moves_mask: Boolean mask of shape (num_actions,) where True = legal move
        
    Returns:
        int: Action index with highest Q-value among legal moves
        
    Raises:
        ValueError: If no legal moves are available
    """
    return epsilon_greedy_action(q_values, legal_moves_mask, epsilon=0.0)


def random_legal_action(legal_moves_mask: torch.Tensor,
                       rng: Optional[random.Random] = None) -> int:
    """
    Select a random legal action (fully random exploration).
    
    Args:
        legal_moves_mask: Boolean mask of shape (num_actions,) where True = legal move
        rng: Optional random number generator for reproducible behavior
        
    Returns:
        int: Randomly selected legal action index
        
    Raises:
        ValueError: If no legal moves are available
    """
    if rng is None:
        rng = random.Random()
    
    legal_indices = torch.nonzero(legal_moves_mask, as_tuple=True)[0]
    if len(legal_indices) == 0:
        raise ValueError("No legal moves available")
    
    random_idx = rng.randint(0, len(legal_indices) - 1)
    return legal_indices[random_idx].item()


def softmax_action(q_values: torch.Tensor,
                  legal_moves_mask: torch.Tensor, 
                  temperature: float = 1.0,
                  rng: Optional[random.Random] = None) -> int:
    """
    Select action using softmax (Boltzmann) exploration with temperature.
    
    Args:
        q_values: Q-values tensor of shape (num_actions,)
        legal_moves_mask: Boolean mask of shape (num_actions,) where True = legal move
        temperature: Temperature parameter (higher = more random, lower = more greedy)
        rng: Optional random number generator for reproducible behavior
        
    Returns:
        int: Selected action index based on softmax probabilities
        
    Raises:
        ValueError: If no legal moves are available
    """
    if rng is None:
        rng = random.Random()
    
    # Get legal move indices
    legal_indices = torch.nonzero(legal_moves_mask, as_tuple=True)[0]
    if len(legal_indices) == 0:
        raise ValueError("No legal moves available")
    
    # Extract Q-values for legal moves only
    legal_q_values = q_values[legal_indices]
    
    # Apply temperature and compute softmax probabilities
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    logits = legal_q_values / temperature
    probs = torch.softmax(logits, dim=0)
    
    # Sample from the probability distribution
    # Convert to numpy for easier random sampling
    probs_np = probs.detach().cpu().numpy()
    
    # Use numpy random choice with our RNG seed
    if hasattr(rng, 'getstate'):
        # Standard random.Random - need to sync with numpy
        np_rng = np.random.RandomState(rng.randint(0, 2**32 - 1))
    else:
        # Fallback
        np_rng = np.random
        
    chosen_legal_idx = np_rng.choice(len(legal_indices), p=probs_np)
    return legal_indices[chosen_legal_idx].item()


def action_to_coordinates(action_idx: int, board_size: int = 15) -> Tuple[int, int]:
    """
    Convert action index to board coordinates.
    
    Args:
        action_idx: Action index (0-224 for 15x15 board)
        board_size: Board size (default 15)
        
    Returns:
        tuple: (row, col) coordinates
    """
    row = action_idx // board_size
    col = action_idx % board_size
    return (row, col)


def coordinates_to_action(row: int, col: int, board_size: int = 15) -> int:
    """
    Convert board coordinates to action index.
    
    Args:
        row: Row coordinate (0-14 for 15x15 board)
        col: Column coordinate (0-14 for 15x15 board)
        board_size: Board size (default 15)
        
    Returns:
        int: Action index
    """
    return row * board_size + col