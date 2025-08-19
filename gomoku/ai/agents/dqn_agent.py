"""
DQN agent for Gomoku.

Integrates state encoding, neural network, and action selection into a complete agent.
"""
import torch
import torch.nn as nn
import random
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from ..models.dqn_model import DuelingDQN
from ..models.state_encoding import encode_game_state, get_legal_moves_mask
from ..models.action_selection import epsilon_greedy_action, action_to_coordinates


class DQNAgent:
    """
    Deep Q-Network agent for Gomoku.
    
    Combines state encoding, neural network inference, and action selection
    to create a complete agent that can play Gomoku games.
    """
    
    def __init__(self, 
                 board_size: int = 15,
                 epsilon: float = 0.1,
                 device: str = 'cpu',
                 seed: Optional[int] = None):
        """
        Initialize DQN agent.
        
        Args:
            board_size: Size of game board (default 15)
            epsilon: Exploration rate for ε-greedy policy (default 0.1)
            device: Device to run model on ('cpu' or 'cuda')
            seed: Random seed for reproducible behavior
        """
        self.board_size = board_size
        self.epsilon = epsilon
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        
        # Initialize neural network
        self.model = DuelingDQN(board_size=board_size)
        self.model.to(self.device)
        
        # Track if model has been trained
        self.is_trained = False
        self.training_info = {
            'episodes': 0,
            'steps': 0,
            'version': '1.0'
        }
    
    def select_action(self, game, last_move: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """
        Select action using current policy.
        
        Compatible with existing agent interface (RandomAgent, HeuristicAgent).
        
        Args:
            game: Game instance with current state
            last_move: Optional last move for state encoding
            
        Returns:
            tuple: (row, col) coordinates of selected move, or None if no legal moves
        """
        # Check for legal moves
        legal_moves = game.board.get_legal_moves()
        if not legal_moves:
            return None
        
        try:
            # Encode game state
            state = encode_game_state(game, last_move=last_move)
            state_batch = state.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Get legal moves mask
            legal_mask = get_legal_moves_mask(game).to(self.device)
            legal_mask_batch = legal_mask.unsqueeze(0)  # Add batch dimension
            
            # Forward pass through network
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state_batch, legal_moves_mask=legal_mask_batch)
                q_values = q_values.squeeze(0)  # Remove batch dimension
            
            # Select action using ε-greedy policy
            action_idx = epsilon_greedy_action(
                q_values.cpu(), 
                legal_mask.cpu(), 
                epsilon=self.epsilon,
                rng=self.rng
            )
            
            # Convert action index to coordinates
            row, col = action_to_coordinates(action_idx, self.board_size)
            return (row, col)
            
        except Exception as e:
            # Fallback to random action if neural network fails
            print(f"DQN forward pass failed: {e}, falling back to random action")
            legal_idx = self.rng.randint(0, len(legal_moves) - 1)
            return legal_moves[legal_idx]
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate."""
        self.epsilon = max(0.0, min(1.0, epsilon))  # Clamp to [0, 1]
    
    def set_evaluation_mode(self, eval_mode: bool = True):
        """
        Set agent to evaluation mode (no exploration) or training mode.
        
        Args:
            eval_mode: If True, set epsilon=0 for pure exploitation
        """
        if eval_mode:
            self._saved_epsilon = self.epsilon
            self.epsilon = 0.0
        else:
            if hasattr(self, '_saved_epsilon'):
                self.epsilon = self._saved_epsilon
    
    def get_q_values(self, game, last_move: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Get Q-values for all actions in current state.
        
        Useful for analysis and debugging.
        
        Args:
            game: Game instance
            last_move: Optional last move for state encoding
            
        Returns:
            torch.Tensor: Q-values of shape (225,)
        """
        state = encode_game_state(game, last_move=last_move)
        state_batch = state.unsqueeze(0).to(self.device)
        
        legal_mask = get_legal_moves_mask(game).to(self.device)
        legal_mask_batch = legal_mask.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_batch, legal_moves_mask=legal_mask_batch)
            return q_values.squeeze(0).cpu()
    
    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save agent model and metadata.
        
        Args:
            filepath: Path to save model (should end in .pt)
            metadata: Optional additional metadata to save
        """
        filepath = Path(filepath)
        
        # Prepare save data
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'board_size': self.board_size,
            'is_trained': self.is_trained,
            'training_info': self.training_info.copy(),
            'model_class': 'DuelingDQN',
        }
        
        if metadata:
            save_data['metadata'] = metadata
        
        # Save model
        torch.save(save_data, filepath)
        
        # Save human-readable metadata
        metadata_path = filepath.with_suffix('.json')
        metadata_info = {
            'model_file': filepath.name,
            'board_size': self.board_size,
            'is_trained': self.is_trained,
            'training_info': self.training_info,
            'model_architecture': 'Dueling Double-DQN',
            'input_channels': 4,
            'output_actions': self.board_size * self.board_size,
        }
        if metadata:
            metadata_info.update(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_info, f, indent=2)
        
        print(f"Model saved to {filepath}")
        print(f"Metadata saved to {metadata_path}")
    
    def load(self, filepath: str):
        """
        Load agent model and metadata.
        
        Args:
            filepath: Path to model file (.pt)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        save_data = torch.load(filepath, map_location=self.device)
        
        # Validate compatibility
        if save_data.get('board_size', 15) != self.board_size:
            raise ValueError(f"Model board size {save_data['board_size']} != agent board size {self.board_size}")
        
        # Load model weights
        self.model.load_state_dict(save_data['model_state_dict'])
        
        # Load metadata
        self.is_trained = save_data.get('is_trained', False)
        self.training_info = save_data.get('training_info', self.training_info)
        
        print(f"Model loaded from {filepath}")
        if self.is_trained:
            episodes = self.training_info.get('episodes', 0)
            steps = self.training_info.get('steps', 0)
            print(f"Model trained for {episodes} episodes, {steps} steps")
        else:
            print("Model is untrained (random weights)")
    
    @classmethod
    def load_from_file(cls, filepath: str, epsilon: float = 0.1, device: str = 'cpu') -> 'DQNAgent':
        """
        Create agent and load model from file.
        
        Args:
            filepath: Path to model file
            epsilon: Exploration rate
            device: Device to run on
            
        Returns:
            DQNAgent: Loaded agent
        """
        # Load basic info to create agent with correct parameters
        save_data = torch.load(filepath, map_location=device)
        board_size = save_data.get('board_size', 15)
        
        # Create agent
        agent = cls(board_size=board_size, epsilon=epsilon, device=device)
        
        # Load the model
        agent.load(filepath)
        
        return agent
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            'type': 'DQNAgent',
            'board_size': self.board_size,
            'epsilon': self.epsilon,
            'is_trained': self.is_trained,
            'training_info': self.training_info,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
        }
    
    def __str__(self) -> str:
        """String representation."""
        status = "trained" if self.is_trained else "untrained"
        return f"DQNAgent(ε={self.epsilon}, {status}, {sum(p.numel() for p in self.model.parameters())} params)"