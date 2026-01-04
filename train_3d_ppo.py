"""
Train PPO on 3D Flight Environment with Obstacles
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from flight_env_3d import FlightControlEnv3D
import argparse
from datetime import datetime


def make_env_3d(env_config, rank=0):
    """Create and wrap 3D environment."""
    def _init():
        env = FlightControlEnv3D(**env_config)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def train_ppo_3d(
    total_timesteps=1000000,  # Artırıldı: 200000 → 1000000 (daha uzun eğitim)
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    world_size=(500.0, 500.0, 200.0),
    num_obstacles=5,
    model_name=None,
    log_dir="./logs_3d/",
    save_dir="./models_3d/",
    eval_freq=20000,
    eval_episodes=5,
    save_freq=50000
):
    """
    Train PPO agent on 3D FlightControlEnv3D.
    """
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Environment configuration - Cruise phase'e göre optimize edilmiş
    env_config = {
        'world_size': world_size,
        'num_obstacles': 1,  # Single fixed obstacle
        'max_episode_steps': 2000,
        'dt': 0.1,
        'max_speed': 30.0,  # Cruise phase'e göre daha da düşürüldü (40 → 30, cruise phase'de hız çok düşük)
        'render_mode': None,
        'use_log_data': True,  # Use FlightGear log data for realistic helicopter behavior
        'log_data_path': 'fg_log2.csv',
        'moving_obstacles': False,  # Fixed obstacles (not moving)
        'obstacle_speed': 0.0,  # No movement
        'target_behind_obstacle': True  # Place target behind the obstacle
    }
    
    # Create environments
    env = DummyVecEnv([make_env_3d(env_config)])
    eval_env = DummyVecEnv([make_env_3d(env_config)])
    
    # Generate model name
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ppo_3d_{timestamp}"
    
    print("=" * 70)
    print("3D PPO TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model Name: {model_name}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"World Size: {world_size}")
    print(f"Number of Obstacles: {num_obstacles}")
    print(f"\n[INFO] Using FlightGear log data: fg_log2.csv")
    print(f"       Episodes will start from realistic states from actual flight data!")
    print(f"       Cruise phase filtering enabled - only level flight data used")
    
    # PPO hyperparameters - Cruise phase'e göre optimize edilmiş
    # Cruise phase'de yumuşak ve stabil davranış öğrenmek için
    ppo_config = {
        'learning_rate': learning_rate if learning_rate != 3e-4 else 5e-5,  # Daha da düşük (1e-4 → 5e-5, daha yavaş ama daha iyi öğrenme)
        'n_steps': n_steps if n_steps != 2048 else 4096,  # Daha uzun (daha fazla deneyim)
        'batch_size': batch_size if batch_size != 64 else 128,  # Daha büyük (daha stabil gradient)
        'n_epochs': n_epochs if n_epochs != 10 else 15,  # Daha fazla (daha iyi öğrenme)
        'gamma': 0.995,  # Daha yüksek (uzun vadeli ödül - cruise phase'de stabilite önemli)
        'gae_lambda': 0.98,  # Daha yüksek (daha smooth advantage - cruise phase'de yumuşak değişimler)
        'clip_range': 0.15,  # Daha küçük (daha konservatif update - cruise phase'de küçük değişimler)
        'ent_coef': 0.02,  # Daha yüksek (daha fazla exploration - cruise phase'de çeşitlilik)
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
        'tensorboard_log': log_dir
    }
    
    print(f"\n[INFO] Cruise phase optimized hyperparameters:")
    print(f"       Learning rate: {ppo_config['learning_rate']}")
    print(f"       N steps: {ppo_config['n_steps']}")
    print(f"       Batch size: {ppo_config['batch_size']}")
    print(f"       N epochs: {ppo_config['n_epochs']}")
    print(f"       Gamma: {ppo_config['gamma']}")
    print(f"       GAE lambda: {ppo_config['gae_lambda']}")
    print(f"       Clip range: {ppo_config['clip_range']}")
    print(f"       Entropy coef: {ppo_config['ent_coef']}")
    print("=" * 70)
    
    # Create PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        **ppo_config
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, f"{model_name}_best"),
        log_path=os.path.join(log_dir, f"{model_name}_eval"),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(save_dir, f"{model_name}_checkpoints"),
        name_prefix=model_name
    )
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_model_path)
    print(f"\n[SUCCESS] Training complete! Model saved to: {final_model_path}")
    print(f"[INFO] TensorBoard logs: {log_dir}")
    print(f"[INFO] Best model: {os.path.join(save_dir, f'{model_name}_best')}")
    
    return model, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for 3D flight control')
    
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total training timesteps (default: 1M for better learning)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--num_obstacles', type=int, default=1,
                       help='Number of obstacles (default: 1 fixed obstacle)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name')
    parser.add_argument('--log_dir', type=str, default='./logs_3d/',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default='./models_3d/',
                       help='Directory for saved models')
    
    args = parser.parse_args()
    
    train_ppo_3d(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_obstacles=args.num_obstacles,
        model_name=args.model_name,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )

