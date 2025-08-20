"""
Main training loop for DQN agent.

Orchestrates self-play data generation, experience replay, and neural network training.
"""
import torch
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..agents.dqn_agent import DQNAgent
from ..agents.random_agent import RandomAgent
from ..agents.heuristic_agent import HeuristicAgent
from ..models.dqn_model import DuelingDQN
from .dqn_trainer import DQNTrainer
from .self_play import SelfPlayManager
from .replay_buffer import ReplayBuffer
from .data_utils import get_default_data_dir


class TrainingConfig:
    """Configuration for DQN training."""
    
    def __init__(self,
                 # Training parameters
                 total_steps: int = 100000,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 grad_clip: float = 1.0,
                 
                 # Experience replay
                 replay_capacity: int = 50000,
                 min_replay_size: int = 1000,
                 
                 # Self-play
                 games_per_update: int = 10,
                 self_play_epsilon: float = 0.3,
                 epsilon_decay_steps: int = 50000,
                 epsilon_min: float = 0.05,
                 
                 # Training schedule
                 train_every: int = 4,
                 target_update_every: int = 1,
                 eval_every: int = 1000,
                 save_every: int = 5000,
                 
                 # Evaluation
                 eval_games: int = 100,
                 
                 # Logging
                 log_every: int = 100,
                 device: str = 'cpu',
                 
                 # Reward structure
                 tactical_rewards: bool = False,
                 
                 # Multi-agent training
                 opponent_weights: Optional[List[float]] = None):
        
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        
        self.replay_capacity = replay_capacity
        self.min_replay_size = min_replay_size
        
        self.games_per_update = games_per_update
        self.self_play_epsilon = self_play_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_min = epsilon_min
        
        self.train_every = train_every
        self.target_update_every = target_update_every
        self.eval_every = eval_every
        self.save_every = save_every
        
        self.eval_games = eval_games
        
        self.log_every = log_every
        self.device = device
        self.tactical_rewards = tactical_rewards
        self.opponent_weights = opponent_weights
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class TrainingLoop:
    """
    Main training loop for DQN agents.
    
    Manages the complete training process including:
    - Self-play data generation
    - Experience replay and neural network training
    - Model evaluation and checkpointing
    - Training progress monitoring
    """
    
    def __init__(self,
                 config: TrainingConfig,
                 data_dir: Optional[str] = None,
                 run_name: Optional[str] = None):
        """
        Initialize training loop.
        
        Args:
            config: Training configuration
            data_dir: Directory for saving data and models
            run_name: Name for this training run
        """
        self.config = config
        
        # Setup directories
        if data_dir is None:
            data_dir = get_default_data_dir()
        else:
            data_dir = Path(data_dir)
        
        if run_name is None:
            run_name = f"dqn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_dir = Path(data_dir) / "training" / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.run_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.device = torch.device(config.device)
        self._setup_agent_and_networks()
        self._setup_training_components()
        self._setup_evaluation_agents()
        
        # Training state
        self.step = 0
        self.episode = 0
        self.training_metrics = []
        self.eval_metrics = []
        
        # Save config
        self._save_config()
        
        print(f"Training run initialized: {run_name}")
        print(f"Run directory: {self.run_dir}")
    
    def _setup_agent_and_networks(self):
        """Setup main agent and networks."""
        # Create main DQN agent
        self.main_agent = DQNAgent(
            board_size=15,
            epsilon=self.config.self_play_epsilon,
            device=self.config.device,
            seed=42
        )
        
        # Extract networks for trainer
        self.online_network = self.main_agent.model
        self.target_network = DuelingDQN(board_size=15).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def _setup_training_components(self):
        """Setup training components."""
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_capacity,
            batch_size=self.config.batch_size
        )
        
        # DQN trainer
        self.trainer = DQNTrainer(
            online_network=self.online_network,
            target_network=self.target_network,
            replay_buffer=self.replay_buffer,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            tau=self.config.tau,
            batch_size=self.config.batch_size,
            grad_clip=self.config.grad_clip,
            device=self.config.device
        )
        
        # Self-play manager
        self.self_play_manager = SelfPlayManager(
            main_agent=self.main_agent,
            games_per_iteration=self.config.games_per_update,
            verbose=False,
            tactical_rewards=self.config.tactical_rewards,
            opponent_weights=self.config.opponent_weights
        )
    
    def _setup_evaluation_agents(self):
        """Setup agents for evaluation."""
        self.eval_agents = {
            'random': RandomAgent(seed=42),
            'heuristic': HeuristicAgent(seed=42)
        }
    
    def _save_config(self):
        """Save training configuration."""
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _update_epsilon(self):
        """Update exploration epsilon with linear decay."""
        if self.step < self.config.epsilon_decay_steps:
            # Linear decay from initial epsilon to min epsilon
            progress = self.step / self.config.epsilon_decay_steps
            current_epsilon = (self.config.self_play_epsilon * (1 - progress) + 
                             self.config.epsilon_min * progress)
        else:
            current_epsilon = self.config.epsilon_min
        
        self.main_agent.set_epsilon(current_epsilon)
        return current_epsilon
    
    def _generate_self_play_data(self) -> int:
        """
        Generate self-play data and add to replay buffer.
        
        Returns:
            int: Number of new transitions added
        """
        # Play games and collect data
        results, transitions = self.self_play_manager.play_games(
            num_games=self.config.games_per_update,
            collect_data=True
        )
        
        # Add to replay buffer
        initial_size = len(self.replay_buffer)
        self.replay_buffer.add_trajectory(transitions)
        new_transitions = len(self.replay_buffer) - initial_size
        
        # Update episode count
        self.episode += len(results)
        
        return new_transitions
    
    def _training_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step.
        
        Returns:
            dict: Training metrics, or None if couldn't train
        """
        if not self.replay_buffer.can_sample(self.config.batch_size):
            return None
        
        # Train on batch
        metrics = self.trainer.train_on_batch()
        
        # Update target network
        if self.step % self.config.target_update_every == 0:
            self.trainer.update_target_network()
        
        return metrics
    
    def _evaluate_agent(self) -> Dict[str, float]:
        """
        Evaluate agent against baseline opponents.
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating agent at step {self.step}...")
        
        # Set agent to evaluation mode (no exploration)
        self.main_agent.set_evaluation_mode(True)
        
        eval_results = {}
        
        for opponent_name, opponent_agent in self.eval_agents.items():
            wins = 0
            draws = 0
            games_played = 0
            
            # Play evaluation games
            for game_idx in range(self.config.eval_games):
                # Alternate colors
                if game_idx % 2 == 0:
                    player1, player2 = self.main_agent, opponent_agent
                    main_is_black = True
                else:
                    player1, player2 = opponent_agent, self.main_agent
                    main_is_black = False
                
                # Use self-play infrastructure for evaluation games
                from .self_play import SelfPlayGame
                eval_game = SelfPlayGame(
                    player1_agent=player1,
                    player2_agent=player2,
                    collect_data=False,
                    verbose=False
                )
                
                result = eval_game.play_game()
                games_played += 1
                
                if result['outcome'] == 'win':
                    if ((main_is_black and result['winner'] == 1) or 
                        (not main_is_black and result['winner'] == -1)):
                        wins += 1
                elif result['outcome'] in ['draw', 'draw_max_moves']:
                    draws += 1
            
            # Calculate metrics
            win_rate = wins / games_played if games_played > 0 else 0.0
            draw_rate = draws / games_played if games_played > 0 else 0.0
            
            eval_results[f'{opponent_name}_win_rate'] = win_rate
            eval_results[f'{opponent_name}_draw_rate'] = draw_rate
            
            print(f"  vs {opponent_name}: {win_rate:.1%} wins, {draw_rate:.1%} draws")
        
        # Restore training mode
        self.main_agent.set_evaluation_mode(False)
        
        return eval_results
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_data = {
            'step': self.step,
            'episode': self.episode,
            'config': self.config.to_dict(),
            'training_metrics': self.training_metrics,
            'eval_metrics': self.eval_metrics,
            'replay_buffer_size': len(self.replay_buffer)
        }
        
        # Save trainer checkpoint
        trainer_path = self.models_dir / f"checkpoint_step_{self.step}.pt"
        self.trainer.save_checkpoint(str(trainer_path), metadata=checkpoint_data)
        
        # Save agent model
        agent_path = self.models_dir / f"agent_step_{self.step}.pt"
        self.main_agent.save(str(agent_path), metadata=checkpoint_data)
        
        # Save best model if this is the best so far
        if is_best:
            best_trainer_path = self.models_dir / "best_trainer.pt"
            best_agent_path = self.models_dir / "best_agent.pt"
            
            # Copy files
            import shutil
            shutil.copy2(trainer_path, best_trainer_path)
            shutil.copy2(agent_path, best_agent_path)
            
            print(f"  Saved as best model!")
        
        print(f"Checkpoint saved at step {self.step}")
    
    def _log_metrics(self, training_metrics: Optional[Dict], eval_metrics: Optional[Dict] = None):
        """Log training metrics."""
        log_entry = {
            'step': self.step,
            'episode': self.episode,
            'epsilon': self.main_agent.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'timestamp': datetime.now().isoformat()
        }
        
        if training_metrics:
            log_entry.update(training_metrics)
        
        if eval_metrics:
            log_entry.update(eval_metrics)
        
        # Store metrics
        self.training_metrics.append(log_entry)
        
        # Save to file
        metrics_file = self.logs_dir / "training_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def train(self) -> Dict[str, List]:
        """
        Run the complete training loop.
        
        Returns:
            dict: Training history with metrics
        """
        print(f"Starting DQN training for {self.config.total_steps} steps")
        print(f"Configuration: {self.config.to_dict()}")
        
        start_time = time.time()
        best_win_rate = 0.0
        
        while self.step < self.config.total_steps:
            step_start_time = time.time()
            
            # Update exploration epsilon
            current_epsilon = self._update_epsilon()
            
            # Generate self-play data
            new_transitions = self._generate_self_play_data()
            
            # Training step
            training_metrics = None
            if (self.step % self.config.train_every == 0 and 
                len(self.replay_buffer) >= self.config.min_replay_size):
                training_metrics = self._training_step()
            
            # Evaluation
            eval_metrics = None
            if self.step % self.config.eval_every == 0 and self.step > 0:
                eval_metrics = self._evaluate_agent()
                
                # Check if this is the best model
                current_win_rate = eval_metrics.get('heuristic_win_rate', 0.0)
                is_best = current_win_rate > best_win_rate
                if is_best:
                    best_win_rate = current_win_rate
            
            # Checkpointing
            if (self.step % self.config.save_every == 0 and self.step > 0) or eval_metrics:
                is_best = eval_metrics and eval_metrics.get('heuristic_win_rate', 0.0) == best_win_rate
                self._save_checkpoint(is_best=is_best)
            
            # Optimized logging - reduce string formatting overhead
            if self.step % self.config.log_every == 0:
                self._log_metrics(training_metrics, eval_metrics)
                
                # Simplified progress reporting for speed
                if self.step % (self.config.log_every * 2) == 0:  # Less frequent detailed logging
                    step_time = time.time() - step_start_time
                    elapsed = time.time() - start_time
                    remaining_steps = self.config.total_steps - self.step
                    eta = (elapsed / (self.step + 1)) * remaining_steps
                    
                    print(f"Step {self.step:6d}/{self.config.total_steps} | "
                          f"ε={current_epsilon:.3f} | "
                          f"Buffer={len(self.replay_buffer):5d} | "
                          f"Episodes={self.episode:4d} | "
                          f"ETA={eta/60:.1f}m")
                    
                    if training_metrics:
                        print(f"  Loss={training_metrics['loss']:.4f} | "
                              f"Q={training_metrics['q_mean']:.3f}±{training_metrics['q_std']:.3f}")
                    
                    if eval_metrics:
                        print(f"  Eval: Random={eval_metrics.get('random_win_rate', 0):.1%} | "
                              f"Heuristic={eval_metrics.get('heuristic_win_rate', 0):.1%}")
                    
                    # Show opponent mix occasionally  
                    if self.step % (self.config.log_every * 8) == 0 and self.config.opponent_weights:
                        print("  ", end="")
                        self.self_play_manager.print_opponent_stats()
                else:
                    # Minimal progress indicator
                    if self.step % self.config.log_every == 0:
                        print(f"Step {self.step}", end=" ", flush=True)
                        if self.step % (self.config.log_every * 10) == 0:
                            print()  # New line every 10 minimal logs
            
            self.step += 1
        
        # Final save
        print("\nTraining completed!")
        final_eval = self._evaluate_agent()
        self._save_checkpoint(is_best=True)  # Save final model
        
        total_time = time.time() - start_time
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Final performance: Random={final_eval.get('random_win_rate', 0):.1%} | "
              f"Heuristic={final_eval.get('heuristic_win_rate', 0):.1%}")
        
        return {
            'training_metrics': self.training_metrics,
            'eval_metrics': self.eval_metrics,
            'config': self.config.to_dict(),
            'run_dir': str(self.run_dir)
        }