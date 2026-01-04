"""
Aşama 1 Eğitimi - Altitude Kontrolü Düzeltilmiş
Helikopterin uçmasını sağlayan özel eğitim
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


class AltitudeFixedEnv(FlightControlEnv3D):
    """
    Altitude kontrolü düzeltilmiş environment.
    Helikopterin uçmasını sağlar.
    """
    
    def step(self, action):
        """Step with fixed altitude control."""
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
        
        # DÜZELTME: Throttle komutunu daha güçlü yap
        # throttle_cmd [-1, 1] -> vertical_accel [düşüş, yükseliş]
        # Helikopterin uçması için: throttle_cmd > 0.5 olmalı
        max_accel = 10.0
        forward_accel = throttle_cmd * max_accel * np.cos(pitch_rad)
        lateral_accel = throttle_cmd * max_accel * np.sin(roll_rad) * np.cos(pitch_rad)
        
        # DÜZELTME: Vertical acceleration - throttle pozitifse yükselsin
        # throttle_cmd = 0.5 -> vertical_accel = 0 (hover)
        # throttle_cmd = 1.0 -> vertical_accel = +10 (yüksel)
        # throttle_cmd = 0.0 -> vertical_accel = -5 (yavaş düşüş)
        # throttle_cmd = -1.0 -> vertical_accel = -15 (hızlı düşüş)
        vertical_accel = (throttle_cmd + 0.5) * 20.0 - 9.81  # Düzeltilmiş!
        
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
        if (self.position[0] < -self.world_size[0]/2 or self.position[0] > self.world_size[0]/2 or
            self.position[1] < -self.world_size[1]/2 or self.position[1] > self.world_size[1]/2 or
            self.position[2] < 2.0 or self.position[2] > self.world_size[2]):
            collision = True
        
        # Calculate reward - ALTITUDE KORUMA ÖNEMLİ
        reward = 0.0
        
        if collision:
            reward += self.collision_penalty
        else:
            # ALTITUDE KORUMA REWARD (ÇOK ÖNEMLİ!)
            target_altitude = 100.0  # Hedef irtifa
            altitude_error = abs(self.position[2] - target_altitude)
            
            # Altitude koruma bonusu
            if altitude_error < 20.0:  # 20m tolerans içinde
                altitude_reward = 5.0 * (1.0 - altitude_error / 20.0)  # Güçlü ödül
                reward += altitude_reward
            elif altitude_error < 50.0:
                altitude_reward = 2.0 * (1.0 - altitude_error / 50.0)
                reward += altitude_reward
            else:
                # Çok düşük veya çok yüksek - penalty
                reward += -1.0 * (altitude_error / 50.0 - 1.0)
            
            # Minimum altitude penalty (düşmeyi önle)
            if self.position[2] < 50.0:  # Çok düşükse
                reward += -10.0 * (1.0 - self.position[2] / 50.0)  # Büyük penalty
            
            # Hedefe ulaşma
            distance_to_target = np.linalg.norm(self.position - self.target_position)
            progress = self.last_distance_to_target - distance_to_target
            
            if progress > 0:
                distance_factor = max(0.3, 1.0 - distance_to_target / 500.0)
                reward += self.progress_reward * progress * 30.0 * distance_factor
            
            self.last_distance_to_target = distance_to_target
            
            # Goal reward
            if distance_to_target < 10.0:
                reward += self.goal_reward
            elif distance_to_target < 50.0:
                proximity_bonus = 50.0 * (1.0 - distance_to_target / 50.0)
                reward += proximity_bonus
            elif distance_to_target < 100.0:
                proximity_bonus = 20.0 * (1.0 - distance_to_target / 100.0)
                reward += proximity_bonus
            
            # Survival bonus
            reward += 0.2  # Her adım hayatta kalma bonusu
        
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
    """Create and wrap 3D environment with altitude fix."""
    def _init():
        env = AltitudeFixedEnv(**env_config)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def train_stage1_altitude_fixed(
    total_timesteps=1500000,  # Daha uzun eğitim
    learning_rate=2e-4,
    model_name=None,
    log_dir="./logs_3d/",
    save_dir="./models_3d/"
):
    """
    Aşama 1 eğitimi - Altitude kontrolü düzeltilmiş.
    """
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"stage1_altitude_fixed_{timestamp}"
    
    print("=" * 70)
    print("ASAMA 1 EGITIMI - ALTITUDE KONTROLU DUZELTILMIS")
    print("=" * 70)
    print(f"Model Adi: {model_name}")
    print(f"Toplam Timesteps: {total_timesteps:,}")
    print(f"\n[OZELLIKLER]")
    print(f"  + Altitude kontrolu duzeltildi - helikopter ucar")
    print(f"  + Altitude koruma reward eklendi")
    print(f"  + Minimum altitude penalty eklendi")
    print(f"  + Engel YOK - sadece hedefe ulasma")
    print("=" * 70)
    
    env_config = {
        'world_size': (500.0, 500.0, 200.0),
        'num_obstacles': 0,  # ENGEL YOK!
        'max_episode_steps': 2000,
        'dt': 0.1,
        'max_speed': 25.0,
        'collision_penalty': -500.0,
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
        save_freq=200000,
        save_path=os.path.join(save_dir, f"{model_name}_checkpoints"),
        name_prefix=model_name
    )
    
    print(f"\n[BASLIYOR] Egitim basliyor...")
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
    parser = argparse.ArgumentParser(description='Aşama 1 eğitimi - Altitude kontrolü düzeltilmiş')
    
    parser.add_argument('--total_timesteps', type=int, default=1500000,
                       help='Toplam eğitim adımı (default: 1.5M)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Öğrenme oranı')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model adı')
    
    args = parser.parse_args()
    
    train_stage1_altitude_fixed(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        model_name=args.model_name
    )

