"""
Aşama 2 Fine-tuning - Son 10m'ye Odaklanma
Mevcut best model'i yükleyip son 10m için fine-tune et
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from train_stage1_optimized import OptimizedTargetReachingEnv, make_env_3d
import argparse
from datetime import datetime


def train_stage2_finetune(
    base_model_path: str,
    total_timesteps=300000,  # Fine-tuning için 300K steps
    learning_rate=1e-4,  # Fine-tuning için düşük learning rate
    model_name=None,
    log_dir="./logs_3d/",
    save_dir="./models_3d/"
):
    """
    Aşama 2 fine-tuning - Mevcut modeli son 10m için fine-tune et.
    """
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model bulunamadı: {base_model_path}")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"stage2_finetune_{timestamp}"
    
    print("=" * 70)
    print("ASAMA 2 FINE-TUNING - SON 10M'YE ODAKLANMA")
    print("=" * 70)
    print(f"Base Model: {base_model_path}")
    print(f"Model Adi: {model_name}")
    print(f"Toplam Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate} (Fine-tuning için düşük)")
    print(f"\n[ASAMA 2 IYILESTIRMELERI]")
    print(f"  1. Son 10-12m icin ultra güçlü bonus (2000 puan)")
    print(f"  2. Progress reward son 10m'de 3x artirildi (60x -> 180x)")
    print(f"  3. Mevcut best model üzerinden fine-tuning")
    print("=" * 70)
    
    # Environment config (Aşama 1 ile aynı)
    env_config = {
        'world_size': (1000.0, 1000.0, 300.0),
        'num_obstacles': 0,
        'max_episode_steps': 5000,
        'dt': 0.1,
        'max_speed': 15.0,
        'collision_penalty': -100.0,
        'obstacle_penalty': 0.0,
        'goal_reward': 2000.0,
        'progress_reward': 3.0,
        'render_mode': None,
        'use_log_data': True,
        'log_data_path': 'fg_log2.csv',
        'moving_obstacles': False,
        'target_behind_obstacle': False
    }
    
    env = DummyVecEnv([make_env_3d(env_config)])
    eval_env = DummyVecEnv([make_env_3d(env_config)])
    
    # Base model'i yükle
    print(f"\n[BILGI] Base model yükleniyor: {base_model_path}")
    model = PPO.load(base_model_path, env=env)
    print("[BASARILI] Base model yüklendi!")
    
    # Fine-tuning için learning rate'i güncelle
    model.learning_rate = learning_rate
    print(f"[BILGI] Learning rate: {learning_rate} (Fine-tuning için)")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, f"{model_name}_best"),
        log_path=os.path.join(log_dir, f"{model_name}_eval"),
        eval_freq=50000,  # Her 50K steps'te evaluation
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Her 50K steps'te checkpoint
        save_path=os.path.join(save_dir, f"{model_name}_checkpoints"),
        name_prefix=model_name
    )
    
    print(f"\n[BASLIYOR] Fine-tuning basliyor...")
    print(f"[BILGI] TensorBoard: tensorboard --logdir {log_dir}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False,  # Mevcut timesteps'i koru
        progress_bar=True
    )
    
    # Final model'i kaydet
    final_model_path = os.path.join(save_dir, f"{model_name}_final.zip")
    model.save(final_model_path)
    print(f"\n[BASARILI] Fine-tuning tamamlandi!")
    print(f"[BILGI] Final model: {final_model_path}")
    print(f"[BILGI] Best model: {os.path.join(save_dir, f'{model_name}_best/best_model.zip')}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aşama 2 Fine-tuning - Son 10m için')
    parser.add_argument('--base_model_path', type=str, 
                       default='./models_3d/stage1_optimized_20260104_020307_best/best_model.zip',
                       help='Base model yolu (Aşama 1 best model)')
    parser.add_argument('--total_timesteps', type=int, default=300000,
                       help='Fine-tuning timesteps (default: 300K)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Fine-tuning learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    train_stage2_finetune(
        base_model_path=args.base_model_path,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate
    )

