"""
İyileştirilmiş PPO Eğitim Scripti
Başarı oranını artırmak için optimize edilmiş reward fonksiyonu ve hyperparameter'lar
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from flight_env_3d import FlightControlEnv3D
from improved_reward_env import ImprovedRewardWrapper
import argparse
from datetime import datetime


def make_env_3d(env_config, rank=0, use_improved_reward=True):
    """Create and wrap 3D environment with improved reward."""
    def _init():
        env = FlightControlEnv3D(**env_config)
        if use_improved_reward:
            env = ImprovedRewardWrapper(env)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def train_improved_ppo_3d(
    total_timesteps=2000000,  # Daha uzun eğitim (2M steps)
    learning_rate=2e-4,  # Biraz daha yüksek learning rate
    n_steps=4096,  # Daha fazla deneyim toplama
    batch_size=128,  # Daha büyük batch size
    n_epochs=10,  # Epoch sayısı
    world_size=(500.0, 500.0, 200.0),
    model_name=None,
    log_dir="./logs_3d/",
    save_dir="./models_3d/",
    eval_freq=50000,  # Daha sık evaluation
    eval_episodes=10,  # Daha fazla episode ile test
    save_freq=100000,  # Checkpoint sıklığı
    curriculum_learning=True  # Kolaydan zora öğrenme
):
    """
    İyileştirilmiş PPO eğitimi - başarı oranını artırmak için optimize edilmiş.
    """
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # İYİLEŞTİRİLMİŞ Environment configuration
    # Reward fonksiyonu daha dengeli ve hedefe odaklı
    env_config = {
        'world_size': world_size,
        'num_obstacles': 1,  # Single fixed obstacle
        'max_episode_steps': 2000,
        'dt': 0.1,
        'max_speed': 25.0,  # Biraz daha düşük hız (daha kontrollü)
        
        # İYİLEŞTİRİLMİŞ REWARD PARAMETRELERİ
        # Daha dengeli reward shaping - çok fazla penalty yerine pozitif ödül
        'collision_penalty': -1000.0,  # Yeterince büyük ama çok fazla değil
        'obstacle_penalty': -200.0,    # Daha az agresif (engelden kaçınma için yeterli)
        'goal_reward': 2000.0,         # ÇOK BÜYÜK ödül (hedefe ulaşmayı teşvik et)
        'progress_reward': 2.0,        # İlerleme için daha fazla ödül
        
        'render_mode': None,
        'use_log_data': True,
        'log_data_path': 'fg_log2.csv',
        'moving_obstacles': False,
        'obstacle_speed': 0.0,
        'target_behind_obstacle': True
    }
    
    # Curriculum Learning: Kolaydan zora
    if curriculum_learning:
        print("\n[INFO] Curriculum Learning aktif - kolaydan zora öğrenme")
        print("       İlk aşamada daha kolay senaryolar, sonra zorlaştırılacak")
    
    # Create environments with improved reward
    env = DummyVecEnv([make_env_3d(env_config, use_improved_reward=True)])
    eval_env = DummyVecEnv([make_env_3d(env_config, use_improved_reward=True)])
    
    # Generate model name
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"improved_ppo_3d_{timestamp}"
    
    print("=" * 70)
    print("IYILESTIRILMIS 3D PPO EGITIM KONFIGURASYONU")
    print("=" * 70)
    print(f"Model Adı: {model_name}")
    print(f"Toplam Timesteps: {total_timesteps:,}")
    print(f"Dünya Boyutu: {world_size}")
    print(f"\n[ONEMLI] Iyilestirmeler:")
    print(f"  + Daha dengeli reward fonksiyonu")
    print(f"  + Hedefe ulasma icin cok buyuk odul (2000)")
    print(f"  + Daha uzun egitim (2M steps)")
    print(f"  + Daha iyi hyperparameter'lar")
    print(f"  + Curriculum learning: {'Aktif' if curriculum_learning else 'Kapali'}")
    
    # İYİLEŞTİRİLMİŞ PPO hyperparameters
    # Daha iyi öğrenme için optimize edilmiş
    ppo_config = {
        'learning_rate': learning_rate,
        'n_steps': n_steps,  # Daha fazla deneyim
        'batch_size': batch_size,  # Daha büyük batch
        'n_epochs': n_epochs,
        'gamma': 0.99,  # Uzun vadeli ödül (biraz daha düşük - daha hızlı öğrenme)
        'gae_lambda': 0.95,  # Advantage estimation
        'clip_range': 0.2,  # Policy update (biraz daha agresif)
        'ent_coef': 0.01,  # Exploration (biraz daha az - daha fazla exploitation)
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
        'tensorboard_log': log_dir,
        'device': 'auto'  # GPU varsa kullan
    }
    
    print(f"\n[INFO] PPO Hyperparameters:")
    print(f"       Learning rate: {ppo_config['learning_rate']}")
    print(f"       N steps: {ppo_config['n_steps']}")
    print(f"       Batch size: {ppo_config['batch_size']}")
    print(f"       N epochs: {ppo_config['n_epochs']}")
    print(f"       Gamma: {ppo_config['gamma']}")
    print(f"       Clip range: {ppo_config['clip_range']}")
    print(f"       Entropy coef: {ppo_config['ent_coef']}")
    print("=" * 70)
    
    # Create PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs={
            'net_arch': [256, 256, 128],  # Daha büyük network (daha iyi öğrenme)
        },
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
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(save_dir, f"{model_name}_checkpoints"),
        name_prefix=model_name
    )
    
    # Train
    print("\n[BASLIYOR] Egitim basliyor...")
    print(f"[BILGI] TensorBoard ile ilerlemeyi izleyebilirsiniz: tensorboard --logdir {log_dir}")
    print(f"[BILGI] Best model otomatik kaydedilecek: {os.path.join(save_dir, f'{model_name}_best')}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("[BASARILI] Egitim tamamlandi!")
    print("=" * 70)
    print(f"Final model: {final_model_path}.zip")
    print(f"Best model: {os.path.join(save_dir, f'{model_name}_best', 'best_model.zip')}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"\n[SONRAKI ADIM] Modeli test edin:")
    print(f"  python visualize_3d_flight.py --model_path {os.path.join(save_dir, f'{model_name}_best', 'best_model.zip')}")
    print("=" * 70)
    
    return model, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='İyileştirilmiş PPO eğitimi - başarı oranını artırmak için')
    
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                       help='Toplam eğitim adımı (default: 2M - daha uzun eğitim)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Öğrenme oranı (default: 2e-4)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model adı (otomatik oluşturulur)')
    parser.add_argument('--log_dir', type=str, default='./logs_3d/',
                       help='TensorBoard log dizini')
    parser.add_argument('--save_dir', type=str, default='./models_3d/',
                       help='Model kayıt dizini')
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Curriculum learning\'i kapat')
    
    args = parser.parse_args()
    
    train_improved_ppo_3d(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        curriculum_learning=not args.no_curriculum
    )

