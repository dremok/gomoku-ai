"""
Dueling Double-DQN network for Gomoku.

Implements the neural network architecture specified in the development plan:
- 4x Conv layers with ReLU + BatchNorm
- Dueling heads: Value (scalar) + Advantage (225 logits)
- Q-value combination: Q = V + (A - mean(A_valid))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling Double-DQN network for Gomoku AI.
    
    Takes 4-channel board state as input and outputs Q-values for all 225 positions.
    Uses dueling architecture to separate state value from action advantages.
    """
    
    def __init__(self, board_size=15, input_channels=4, conv_filters=64):
        """
        Initialize the Dueling DQN network.
        
        Args:
            board_size: Size of the game board (default 15 for 15x15)
            input_channels: Number of input channels (default 4)
            conv_filters: Number of filters in convolutional layers (default 64)
        """
        super(DuelingDQN, self).__init__()
        
        self.board_size = board_size
        self.input_channels = input_channels
        self.conv_filters = conv_filters
        self.num_actions = board_size * board_size  # 225 for 15x15 board
        
        # Convolutional feature extractor
        # 4 conv layers as specified in development plan
        self.conv1 = nn.Conv2d(input_channels, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        
        self.conv2 = nn.Conv2d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters)
        
        self.conv3 = nn.Conv2d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters)
        
        self.conv4 = nn.Conv2d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_filters)
        
        # Calculate the size of flattened features after conv layers
        # Since we use padding=1 with kernel_size=3, spatial dimensions are preserved
        self.conv_output_size = conv_filters * board_size * board_size
        
        # Dueling heads
        # Value head: estimates state value V(s)
        self.value_head = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single scalar value
        )
        
        # Advantage head: estimates action advantages A(s,a)
        self.advantage_head = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)  # One value per action
        )
    
    def forward(self, x, legal_moves_mask=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 4, 15, 15)
            legal_moves_mask: Optional boolean mask of shape (batch, 225)
                            where True = legal move, False = illegal move
                            
        Returns:
            torch.Tensor: Q-values of shape (batch, 225)
                         Illegal moves will have very negative Q-values if mask provided
        """
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten for fully connected layers
        x = x.view(batch_size, -1)
        
        # Dueling heads
        value = self.value_head(x)  # Shape: (batch, 1)
        advantage = self.advantage_head(x)  # Shape: (batch, 225)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        # This ensures that the value function is identifiable
        if legal_moves_mask is not None:
            # Only consider advantages of legal moves for the mean calculation
            # Set advantages of illegal moves to 0 for mean calculation
            masked_advantage = advantage * legal_moves_mask.float()
            legal_count = legal_moves_mask.sum(dim=1, keepdim=True).float()
            # Avoid division by zero
            legal_count = torch.clamp(legal_count, min=1.0)
            advantage_mean = masked_advantage.sum(dim=1, keepdim=True) / legal_count
        else:
            # Use all advantages for mean calculation
            advantage_mean = advantage.mean(dim=1, keepdim=True)
        
        # Combine value and normalized advantage
        q_values = value + advantage - advantage_mean
        
        # Apply legal moves masking by setting illegal moves to very negative values
        if legal_moves_mask is not None:
            # Set illegal moves to a very negative value so they're never selected
            illegal_penalty = -1e6
            q_values = q_values.masked_fill(~legal_moves_mask, illegal_penalty)
        
        return q_values
    
    def get_feature_size(self):
        """Get the size of the flattened convolutional features."""
        return self.conv_output_size