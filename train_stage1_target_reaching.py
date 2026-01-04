"""
Aşama 1 Eğitimi - Hedefe Ulaşma Optimizasyonu
Boundary penalty azaltıldı, reward fine-tuning yapıldı
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


class TargetReachingEnv(FlightControlEnv3D):
    """
    Hedefe ulaşma için optimize edilmiş environment.
    Boundary penalty azaltıldı, reward fine-tuning yapıldı.
    """
    
    def step(self, action):
        """Step with optimized reward for target reaching."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = action
        
        # Update attitude
        attitude_update_rate = 5.0
        self.attitude[0] += roll_cmd * attitude_update_rate * self.dt
        self.attitude[1] += pitch_cmd * attitude_update_rate * self.dt
        self.attitude[2] += yaw_cmd * 15.0 * self.dt
        
        self.attitude[0] = np.clip(self.attitude[0], -90.0, 90.0)
        self.attitude[1] = np.clip(self.attitude[1], -90.0, 90.0)
        self.attitude[2] = self.attitude[2] % 360.0
        
        roll_rad = np.deg2rad(self.attitude[0])
        pitch_rad = np.deg2rad(self.attitude[1])
        yaw_rad = np.deg2rad(self.attitude[2])
        
        # Düzeltilmiş throttle kontrolü
        max_accel = 10.0
        forward_accel = throttle_cmd * max_accel * np.cos(pitch_rad)
        lateral_accel = throttle_cmd * max_accel * np.sin(roll_rad) * np.cos(pitch_rad)
        vertical_accel = (throttle_cmd + 0.5) * 20.0 - 9.81
        
        # Update velocity
        self.velocity[0] += forward_accel * np.cos(yaw_rad) * self.dt
        self.velocity[1] += forward_accel * np.sin(yaw_rad) * self.dt
        self.velocity[0] += lateral_accel * np.cos(yaw_rad + np.pi/2) * self.dt
        self.velocity[1] += lateral_accel * np.sin(yaw_rad + np.pi/2) * self.dt
        self.velocity[2] += vertical_accel * self.dt
        
        # Velocity damping
        damping_factor = 0.98
        self.velocity *= damping_factor
        
        # Clamp velocity
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        # Update position
        self.position += self.velocity * self.dt
        
        # Update altitude rate
        self.altitude_rate = self.velocity[2]
        
        # Check collisions (sadece boundary)
        collision = False
        boundary_penalty = 0.0
        
        # Boundary kontrolü - AZALTILMIŞ PENALTY
        world_half_x = self.world_size[0] / 2
        world_half_y = self.world_size[1] / 2
        world_max_z = self.world_size[2]
        
        # Boundary'ye yaklaşırsa uyarı (penalty değil, sadece uyarı)
        boundary_warning = False
        if (abs(self.position[0]) > world_half_x * 0.8 or 
            abs(self.position[1]) > world_half_y * 0.8 or
            self.position[2] < 20.0 or self.position[2] > world_max_z * 0.9):
            boundary_warning = True
        
        # Sadece gerçek boundary collision'da penalty
        if (self.position[0] < -world_half_x or self.position[0] > world_half_x or
            self.position[1] < -world_half_y or self.position[1] > world_half_y or
            self.position[2] < 2.0 or self.position[2] > world_max_z):
            collision = True
            boundary_penalty = -200.0  # AZALTILDI: -500 → -200
        
        # Calculate reward - HEDEFE ULAŞMA ODAKLI
        reward = 0.0
        
        if collision:
            reward += boundary_penalty  # Azaltılmış penalty
        else:
            # ALTITUDE KORUMA REWARD
            target_altitude = 100.0
            altitude_error = abs(self.position[2] - target_altitude)
            
            if altitude_error < 20.0:
                altitude_reward = 5.0 * (1.0 - altitude_error / 20.0)
                reward += altitude_reward
            elif altitude_error < 50.0:
                altitude_reward = 2.0 * (1.0 - altitude_error / 50.0)
                reward += altitude_reward
            else:
                reward += -1.0 * (altitude_error / 50.0 - 1.0)
            
            # Minimum altitude penalty
            if self.position[2] < 50.0:
                reward += -10.0 * (1.0 - self.position[2] / 50.0)
            
            # BOUNDARY'DEN UZAK DURMA BONUSU (YENİ!)
            # Merkeze yakınsa bonus
            center_distance = np.sqrt(
                (self.position[0] / world_half_x) ** 2 +
                (self.position[1] / world_half_y) ** 2
            )
            if center_distance < 0.5:  # Merkeze yakınsa
                boundary_safety_bonus = 0.5 * (1.0 - center_distance / 0.5)
                reward += boundary_safety_bonus
            
            # Boundary uyarısı varsa küçük penalty
            if boundary_warning:
                reward += -0.5  # Küçük uyarı penalty
            
            # HEDEFE ULAŞMA - ÇOK GÜÇLENDİRİLDİ
            distance_to_target = np.linalg.norm(self.position - self.target_position)
            progress = self.last_distance_to_target - distance_to_target
            
            # İlerleme ödülü - DAHA GÜÇLÜ
            if progress > 0:
                distance_factor = max(0.5, 1.0 - distance_to_target / 500.0)
                progress_reward = progress * self.progress_reward * 50.0 * distance_factor  # 30 → 50
                reward += progress_reward
            
            self.last_distance_to_target = distance_to_target
            
            # Goal reward - ÇOK GÜÇLENDİRİLDİ
            if distance_to_target < 10.0:  # Hedefe ulaşıldı!
                reward += self.goal_reward * 2.0  # 2000 → 4000 puan!
                # Ekstra bonus
                proximity_bonus = 200.0 * (1.0 - distance_to_target / 10.0)
                reward += proximity_bonus
            elif distance_to_target < 20.0:  # Çok yakın
                proximity_bonus = 100.0 * (1.0 - distance_to_target / 20.0)
                reward += proximity_bonus
            elif distance_to_target < 50.0:  # Yakın
                proximity_bonus = 50.0 * (1.0 - distance_to_target / 50.0)
                reward += proximity_bonus
            elif distance_to_target < 100.0:  # Orta mesafe
                proximity_bonus = 20.0 * (1.0 - distance_to_target / 100.0)
                reward += proximity_bonus
            
            # Survival bonus - artırıldı
            reward += 0.3  # 0.2 → 0.3
        
        self.episode_rewards.append(reward)
        self.step_count += 1
        
        terminated = collision
        truncated = (self.step_count >= self.max_episode_steps)
        
        info = {
            'position': self.position.copy(),
            'target': self.target_position.copy(),
            'distance_to_target': self.last_distance_to_target,
            'collision': collision,
            'altitude': self.position[2]
        }
        
        return self._get_state(), reward, terminated, truncated, info


def make_env_3d(env_config, rank=0):
    """Create and wrap 3D environment with target reaching optimization."""
    def _init():
        env = TargetReachingEnv(**env_config)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def train_stage1_target_reaching(
    total_timesteps=2500000,  # DAHA UZUN: 2.5M steps
    learning_rate=2e-4,
    model_name=None,
    log_dir="./logs_3d/",
    save_dir="./models_3d/"
):
    """
    Aşama 1 eğitimi - Hedefe ulaşma optimizasyonu.
    """
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"stage1_target_reaching_{timestamp}"
    
    print("=" * 70)
    print("ASAMA 1 EGITIMI - HEDEFE ULASMA OPTIMIZASYONU")
    print("=" * 70)
    print(f"Model Adi: {model_name}")
    print(f"Toplam Timesteps: {total_timesteps:,}")
    print(f"\n[UYGULANAN IYILESTIRMELER]")
    print(f"  + Boundary penalty azaltildi: -500 -> -200")
    print(f"  + Boundary'den uzak durma bonusu eklendi")
    print(f"  + Hedefe ulasma reward cok guclendirildi (4000 puan!)")
    print(f"  + Ilerleme reward artirildi (50x multiplier)")
    print(f"  + Daha uzun egitim: 2.5M steps")
    print(f"  + Survival bonus artirildi")
    print("=" * 70)
    
    env_config = {
        'world_size': (500.0, 500.0, 200.0),
        'num_obstacles': 0,  # ENGEL YOK!
        'max_episode_steps': 2000,
        'dt': 0.1,
        'max_speed': 25.0,
        'collision_penalty': -200.0,  # AZALTILDI: -500 → -200
        'obstacle_penalty': 0.0,
        'goal_reward': 2000.0,  # Base reward (final'de 2x olacak = 4000)
        'progress_reward': 3.0,
        'render_mode': None,
        'use_log_data': True,
        'log_data_path': 'fg_log2.csv',
        'moving_obstacles': False,
        'target_behind_obstacle': False
    }
    
    env = DummyVecEnv([make_env_3d(env_config)])
    eval_env = DummyVecEnv([make_env_3d(env_config)])
    
    ppo_config = {
        'learning_rate': learning_rate,
        'n_steps': 4096,
        'batch_size': 128,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
        'tensorboard_log': log_dir,
        'device': 'auto',
        'policy_kwargs': {
            'net_arch': [256, 256, 128]
        }
    }
    
    model = PPO('MlpPolicy', env, **ppo_config)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, f"{model_name}_best"),
        log_path=os.path.join(log_dir, f"{model_name}_eval"),
        eval_freq=50000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=250000,  # Her 250K steps'te bir
        save_path=os.path.join(save_dir, f"{model_name}_checkpoints"),
        name_prefix=model_name
    )
    
    print(f"\n[BASLIYOR] Egitim basliyor...")
    print(f"[BILGI] TensorBoard: tensorboard --logdir {log_dir}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    final_model_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("[BASARILI] Egitim tamamlandi!")
    print("=" * 70)
    print(f"Final model: {final_model_path}.zip")
    print(f"Best model: {os.path.join(save_dir, f'{model_name}_best', 'best_model.zip')}")
    print(f"\n[TEST] Modeli test edin:")
    print(f"  python test_stage1_no_obstacle.py --model_path {os.path.join(save_dir, f'{model_name}_best', 'best_model.zip')}")
    print("=" * 70)
    
    return model, model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aşama 1 eğitimi - Hedefe ulaşma optimizasyonu')
    
    parser.add_argument('--total_timesteps', type=int, default=2500000,
                       help='Toplam eğitim adımı (default: 2.5M)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Öğrenme oranı')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model adı')
    
    args = parser.parse_args()
    
    train_stage1_target_reaching(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        model_name=args.model_name
    )

