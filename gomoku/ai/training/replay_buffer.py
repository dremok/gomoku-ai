"""
Replay buffer for storing and sampling game experiences for DQN training.

The replay buffer stores transitions (state, action, reward, next_state, done)
and provides efficient sampling for training the neural network.
"""
import torch
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Iterator
import pickle
import gzip
from pathlib import Path


# Define transition structure for clarity
Transition = namedtuple('Transition', [
    'state',      # Current game state (4-channel tensor)
    'action',     # Action taken (action index)
    'reward',     # Immediate reward
    'next_state', # Next game state (4-channel tensor)
    'done'        # Whether episode ended
])


class ReplayBuffer:
    """
    Circular replay buffer for storing game experiences.
    
    Efficiently stores transitions and provides random sampling for training.
    Uses fixed-size circular buffer to avoid memory growth.
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 batch_size: int = 32,
                 seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Default batch size for sampling
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        
        # Track statistics
        self.total_added = 0
        
    def add(self, state: torch.Tensor, action: int, reward: float, 
            next_state: Optional[torch.Tensor], done: bool):
        """
        Add a single transition to the buffer.
        
        Args:
            state: Current state tensor of shape (4, H, W)
            action: Action index taken
            reward: Immediate reward received
            next_state: Next state tensor, None if episode ended
            done: True if episode ended
        """
        # Convert to CPU tensors to save GPU memory
        state = state.cpu()
        next_state = next_state.cpu() if next_state is not None else None
        
        transition = Transition(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=bool(done)
        )
        
        self.buffer.append(transition)
        self.total_added += 1
    
    def add_trajectory(self, trajectory: List[Transition]):
        """
        Add multiple transitions from a complete game trajectory.
        
        Args:
            trajectory: List of transitions from one game
        """
        for transition in trajectory:
            self.add(
                transition.state,
                transition.action, 
                transition.reward,
                transition.next_state,
                transition.done
            )
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample (default: self.batch_size)
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones) as tensors
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        # Sample random transitions
        transitions = self.rng.sample(self.buffer, batch_size)
        
        # Separate the batch into components
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for t in transitions:
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)
            
            if t.next_state is not None:
                next_states.append(t.next_state)
            else:
                # Create zero tensor for terminal states
                next_states.append(torch.zeros_like(t.state))
            
            dones.append(t.done)
        
        # Stack into batch tensors
        states_batch = torch.stack(states)
        actions_batch = torch.tensor(actions, dtype=torch.long)
        rewards_batch = torch.tensor(rewards, dtype=torch.float32)
        next_states_batch = torch.stack(next_states)
        dones_batch = torch.tensor(dones, dtype=torch.bool)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
    
    def can_sample(self, batch_size: Optional[int] = None) -> bool:
        """Check if buffer has enough samples for a batch."""
        if batch_size is None:
            batch_size = self.batch_size
        return len(self.buffer) >= batch_size
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) == self.capacity
    
    def clear(self):
        """Clear all stored transitions."""
        self.buffer.clear()
        self.total_added = 0
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'utilization': len(self.buffer) / self.capacity,
            'is_full': self.is_full()
        }
    
    def save(self, filepath: str):
        """
        Save replay buffer to disk.
        
        Args:
            filepath: Path to save file (will add .gz compression)
        """
        filepath = Path(filepath)
        if not filepath.suffix == '.gz':
            filepath = filepath.with_suffix(filepath.suffix + '.gz')
        
        # Convert buffer to list for pickling
        data = {
            'buffer': list(self.buffer),
            'capacity': self.capacity,
            'batch_size': self.batch_size,
            'total_added': self.total_added
        }
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Replay buffer saved to {filepath} ({len(self.buffer)} transitions)")
    
    def load(self, filepath: str):
        """
        Load replay buffer from disk.
        
        Args:
            filepath: Path to saved buffer file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Replay buffer file not found: {filepath}")
        
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore buffer state
        self.capacity = data['capacity']
        self.batch_size = data['batch_size']
        self.total_added = data['total_added']
        
        # Rebuild deque with correct capacity
        self.buffer = deque(data['buffer'], maxlen=self.capacity)
        
        print(f"Replay buffer loaded from {filepath} ({len(self.buffer)} transitions)")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ReplayBuffer':
        """
        Create new replay buffer and load from file.
        
        Args:
            filepath: Path to saved buffer file
            
        Returns:
            ReplayBuffer: Loaded buffer instance
        """
        buffer = cls(capacity=1)  # Temporary, will be overwritten
        buffer.load(filepath)
        return buffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer that samples important transitions more frequently.
    
    Uses TD-error based priorities to focus learning on more informative experiences.
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 batch_size: int = 32,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Default batch size for sampling
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (annealed to 1.0)
            beta_increment: How much to increase beta per sample
            seed: Random seed for reproducible sampling
        """
        super().__init__(capacity, batch_size, seed)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Priority storage (same size as buffer)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(self, state: torch.Tensor, action: int, reward: float,
            next_state: Optional[torch.Tensor], done: bool, priority: Optional[float] = None):
        """
        Add transition with priority.
        
        Args:
            state: Current state tensor
            action: Action index taken
            reward: Immediate reward received
            next_state: Next state tensor, None if episode ended
            done: True if episode ended
            priority: TD-error based priority (uses max if None)
        """
        super().add(state, action, reward, next_state, done)
        
        # Add priority (use max priority for new experiences)
        if priority is None:
            priority = self.max_priority
        
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions that were sampled
            priorities: New priorities (typically TD-errors)
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(priority) + 1e-6  # Small epsilon for stability
                self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones, weights, indices)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        # Convert priorities to probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample according to priorities
        indices = self.rng.choices(range(len(self.buffer)), weights=probabilities, k=batch_size)
        
        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by max weight
        
        # Prepare batch tensors (same as parent class)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for t in transitions:
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)
            
            if t.next_state is not None:
                next_states.append(t.next_state)
            else:
                next_states.append(torch.zeros_like(t.state))
            
            dones.append(t.done)
        
        # Stack into batch tensors
        states_batch = torch.stack(states)
        actions_batch = torch.tensor(actions, dtype=torch.long)
        rewards_batch = torch.tensor(rewards, dtype=torch.float32)
        next_states_batch = torch.stack(next_states)
        dones_batch = torch.tensor(dones, dtype=torch.bool)
        weights_batch = torch.tensor(weights, dtype=torch.float32)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, weights_batch, indices