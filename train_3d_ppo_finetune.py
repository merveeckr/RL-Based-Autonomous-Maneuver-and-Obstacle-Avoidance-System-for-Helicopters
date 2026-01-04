"""
Fine-tune existing model for navigation while preserving similarity.
Loads cruise_optimized_v2 model and fine-tunes it for obstacle avoidance and goal reaching.
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


def finetune_model(
    base_model_path: str,
    total_timesteps: int = 300000,
    learning_rate: float = 1e-5,  # Düşük learning rate (fine-tuning için)
    model_name: str = None,
    log_dir: str = "./logs_3d/",
    save_dir: str = "./models_3d/",
    eval_freq: int = 20000,
    eval_episodes: int = 5,
    save_freq: int = 50000
):
    """
    Fine-tune existing model for navigation while preserving similarity.
    
    Args:
        base_model_path: Path to base model (cruise_optimized_v2)
        total_timesteps: Total fine-tuning timesteps
        learning_rate: Learning rate for fine-tuning (should be low)
        model_name: Name for fine-tuned model
        log_dir: Directory for TensorBoard logs
        save_dir: Directory for saved models
    """
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Environment configuration - Navigation odaklı ama stability koru
    env_config = {
        'world_size': (500.0, 500.0, 200.0),
        'num_obstacles': 1,  # Single fixed obstacle
        'max_episode_steps': 2000,
        'dt': 0.1,
        'max_speed': 30.0,  # Cruise phase'e göre
        'collision_penalty': -500.0,  # Çok artırıldı: collision'ı kesinlikle önlemek için
        'obstacle_penalty': -200.0,   # Artırıldı: navigation için
        'goal_reward': 500.0,         # Çok artırıldı: hedefe ulaşmayı ödüllendirmek için
        'progress_reward': 0.5,       # Artırıldı: navigation için
        'render_mode': None,
        'use_log_data': True,  # Cruise phase log data kullan
        'log_data_path': 'fg_log2.csv',
        'moving_obstacles': False,
        'obstacle_speed': 0.0,
        'target_behind_obstacle': True
    }
    
    # Create environments
    env = DummyVecEnv([make_env_3d(env_config)])
    eval_env = DummyVecEnv([make_env_3d(env_config)])
    
    # Generate model name
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"cruise_nav_finetuned_{timestamp}"
    
    print("=" * 70)
    print("FINE-TUNING FOR NAVIGATION (Preserving Similarity)")
    print("=" * 70)
    print(f"Base Model: {base_model_path}")
    print(f"Fine-tuned Model Name: {model_name}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"\n[INFO] Navigation reward'ları çok artırıldı:")
    print(f"       Collision penalty: -500.0")
    print(f"       Obstacle penalty: -200.0")
    print(f"       Goal reward: 500.0")
    print(f"       Progress reward: 0.5")
    print(f"\n[INFO] Stability reward'ları korunuyor (benzerlik için)")
    print("=" * 70)
    
    # Load base model
    print(f"\n[INFO] Loading base model: {base_model_path}")
    model = PPO.load(base_model_path, env=env)
    print("[OK] Base model loaded!")
    
    # Update learning rate for fine-tuning
    model.learning_rate = learning_rate
    print(f"[OK] Learning rate set to: {learning_rate}")
    
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
    
    # Fine-tune (reset_num_timesteps=False to preserve timesteps)
    print("\n[INFO] Starting fine-tuning...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Mevcut timesteps'i koru
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_model_path)
    print(f"\n[SUCCESS] Fine-tuning complete! Model saved to: {final_model_path}")
    print(f"[INFO] TensorBoard logs: {log_dir}")
    print(f"[INFO] Best model: {os.path.join(save_dir, f'{model_name}_best')}")
    
    return model, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune model for navigation')
    parser.add_argument('--base_model', type=str, 
                       default='./models_3d/cruise_optimized_v2_best/best_model.zip',
                       help='Path to base model to fine-tune')
    parser.add_argument('--total_timesteps', type=int, default=300000,
                       help='Total fine-tuning timesteps')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for fine-tuned model')
    parser.add_argument('--log_dir', type=str, default='./logs_3d/',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default='./models_3d/',
                       help='Directory for saved models')
    
    args = parser.parse_args()
    
    finetune_model(
        base_model_path=args.base_model,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )

