"""
DQN trainer for Gomoku AI.

Implements Double DQN with target network, experience replay, and training optimization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from ..models.dqn_model import DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNTrainer:
    """
    Deep Q-Network trainer with Double DQN and target network.
    
    Implements the training algorithm for DQN agents with:
    - Double DQN for reduced overestimation bias
    - Target network for stable Q-learning
    - Experience replay for sample efficiency
    - Huber loss for robust training
    """
    
    def __init__(self,
                 online_network: DuelingDQN,
                 target_network: Optional[DuelingDQN] = None,
                 replay_buffer: Optional[ReplayBuffer] = None,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 batch_size: int = 32,
                 grad_clip: float = 1.0,
                 device: str = 'cpu',
                 double_dqn: bool = True):
        """
        Initialize DQN trainer.
        
        Args:
            online_network: Online (training) network
            target_network: Target network (if None, creates copy of online)
            replay_buffer: Experience replay buffer
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            tau: Soft update rate for target network (0.01 = 1% update per step)
            batch_size: Training batch size
            grad_clip: Gradient clipping value
            device: Device to run training on
            double_dqn: Whether to use Double DQN algorithm
        """
        self.device = torch.device(device)
        
        # Networks
        self.online_network = online_network.to(self.device)
        if target_network is None:
            self.target_network = copy.deepcopy(online_network).to(self.device)
        else:
            self.target_network = target_network.to(self.device)
        
        # Training parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.double_dqn = double_dqn
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = replay_buffer
        
        # Training statistics
        self.training_steps = 0
        self.total_loss = 0.0
        self.q_value_stats = {'mean': 0.0, 'std': 0.0}
        
        # Set target network to eval mode (no gradient computation needed)
        self.target_network.eval()
        
    def compute_td_targets(self, 
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          next_legal_masks: torch.Tensor,
                          dones: torch.Tensor) -> torch.Tensor:
        """
        Compute TD targets for Q-learning.
        
        Uses Double DQN: action selection with online network, Q-value with target network.
        
        Args:
            rewards: Batch of rewards, shape (batch_size,)
            next_states: Batch of next states, shape (batch_size, 4, 15, 15)
            next_legal_masks: Legal moves for next states, shape (batch_size, 225)
            dones: Whether episodes ended, shape (batch_size,)
            
        Returns:
            torch.Tensor: TD targets, shape (batch_size,)
        """
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: select actions with online network
                online_next_q = self.online_network(next_states, next_legal_masks)
                
                # Mask out illegal moves for action selection
                masked_online_q = online_next_q.clone()
                masked_online_q[~next_legal_masks] = float('-inf')
                
                # Select best actions according to online network
                next_actions = torch.argmax(masked_online_q, dim=1)
                
                # Evaluate selected actions with target network
                target_next_q = self.target_network(next_states, next_legal_masks)
                next_q_values = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: select and evaluate with target network
                target_next_q = self.target_network(next_states, next_legal_masks)
                
                # Mask out illegal moves
                masked_target_q = target_next_q.clone()
                masked_target_q[~next_legal_masks] = float('-inf')
                
                next_q_values = torch.max(masked_target_q, dim=1)[0]
            
            # Compute TD targets: r + γ * max_a Q_target(s', a) * (1 - done)
            td_targets = rewards + self.gamma * next_q_values * (~dones).float()
            
        return td_targets
    
    def train_step(self, 
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   rewards: torch.Tensor,
                   next_states: torch.Tensor,
                   dones: torch.Tensor,
                   legal_masks: torch.Tensor,
                   next_legal_masks: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            states: Current states, shape (batch_size, 4, 15, 15)
            actions: Actions taken, shape (batch_size,)
            rewards: Rewards received, shape (batch_size,)
            next_states: Next states, shape (batch_size, 4, 15, 15)
            dones: Episode termination flags, shape (batch_size,)
            legal_masks: Legal moves for current states, shape (batch_size, 225)
            next_legal_masks: Legal moves for next states, shape (batch_size, 225)
            
        Returns:
            dict: Training metrics (loss, q_mean, q_std)
        """
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        legal_masks = legal_masks.to(self.device)
        next_legal_masks = next_legal_masks.to(self.device)
        
        # Forward pass through online network
        self.online_network.train()
        current_q_values = self.online_network(states, legal_masks)
        
        # Get Q-values for taken actions
        action_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute TD targets
        td_targets = self.compute_td_targets(rewards, next_states, next_legal_masks, dones)
        
        # Compute loss
        loss = self.loss_fn(action_q_values, td_targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update statistics
        self.training_steps += 1
        self.total_loss += loss.item()
        
        # Q-value statistics
        with torch.no_grad():
            q_mean = current_q_values.mean().item()
            q_std = current_q_values.std().item()
            self.q_value_stats = {'mean': q_mean, 'std': q_std}
        
        return {
            'loss': loss.item(),
            'q_mean': q_mean,
            'q_std': q_std,
            'td_targets_mean': td_targets.mean().item(),
            'td_targets_std': td_targets.std().item()
        }
    
    def train_on_batch(self) -> Optional[Dict[str, float]]:
        """
        Train on a single batch from the replay buffer.
        
        Returns:
            dict: Training metrics, or None if not enough samples in buffer
        """
        if self.replay_buffer is None:
            raise ValueError("No replay buffer provided")
        
        if not self.replay_buffer.can_sample(self.batch_size):
            return None
        
        # Sample batch from replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # Prioritized replay returns additional items
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, weights, indices = batch
            
            # TODO: Use importance sampling weights for prioritized replay
            # For now, treat as uniform sampling
        else:
            # Standard replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Create legal move masks for current and next states
        # Optimized: use device tensors and avoid unnecessary operations
        batch_size = states.shape[0]
        legal_masks = torch.ones(batch_size, 225, dtype=torch.bool, device=states.device)
        next_legal_masks = torch.ones(batch_size, 225, dtype=torch.bool, device=states.device)
        
        # For terminal states, next_legal_masks should be all False (vectorized)
        next_legal_masks[dones] = False
        
        return self.train_step(states, actions, rewards, next_states, dones, 
                             legal_masks, next_legal_masks)
    
    def update_target_network(self):
        """
        Update target network using soft update (Polyak averaging).
        
        θ_target = τ * θ_online + (1 - τ) * θ_target
        """
        for target_param, online_param in zip(self.target_network.parameters(), 
                                            self.online_network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def hard_update_target_network(self):
        """
        Hard update: copy online network weights to target network.
        """
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def get_training_stats(self) -> Dict[str, float]:
        """
        Get current training statistics.
        
        Returns:
            dict: Training statistics
        """
        avg_loss = self.total_loss / max(self.training_steps, 1)
        
        return {
            'training_steps': self.training_steps,
            'avg_loss': avg_loss,
            'total_loss': self.total_loss,
            'q_mean': self.q_value_stats['mean'],
            'q_std': self.q_value_stats['std'],
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            metadata: Additional metadata to save
        """
        checkpoint = {
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'total_loss': self.total_loss,
            'q_value_stats': self.q_value_stats,
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'grad_clip': self.grad_clip,
                'double_dqn': self.double_dqn
            }
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            dict: Loaded metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.training_steps = checkpoint.get('training_steps', 0)
        self.total_loss = checkpoint.get('total_loss', 0.0)
        self.q_value_stats = checkpoint.get('q_value_stats', {'mean': 0.0, 'std': 0.0})
        
        return checkpoint.get('metadata', {})
    
    def set_learning_rate(self, lr: float):
        """Set optimizer learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr