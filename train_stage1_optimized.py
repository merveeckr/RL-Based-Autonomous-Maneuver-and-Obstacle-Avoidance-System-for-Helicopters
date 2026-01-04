"""
Aşama 1 Eğitimi - Optimize Edilmiş Versiyon
Tüm iyileştirmeler uygulandı: Dünya büyütüldü, başlangıç merkeze alındı, hız azaltıldı
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


class OptimizedTargetReachingEnv(FlightControlEnv3D):
    """
    Optimize edilmiş environment - tüm iyileştirmeler uygulandı.
    """
    
    def reset(self, seed=None, options=None):
        """Reset with FIXED initial positions (center of world)."""
        if seed is not None:
            np.random.seed(seed)
        
        # SABİT BAŞLANGIÇ POZİSYONU - MERKEZE YAKIN
        # Her episode'da aynı yerde başla
        self.position = np.array([0.0, 0.0, 100.0], dtype=np.float32)  # Tam merkez, 100m irtifa
        
        # Log data kullanılıyorsa attitude ve velocity'yi log'dan al
        if self.use_log_data and self.log_states is not None and len(self.log_states) > 0:
            idx = np.random.randint(0, len(self.log_states))
            log_state = self.log_states[idx]
            
            # Attitude from log
            self.attitude = np.array([
                log_state[1],  # roll
                log_state[2],  # pitch
                log_state[3]   # heading (yaw)
            ], dtype=np.float32)
            
            # Velocity from log
            vz = log_state[4]  # altitude_rate
            heading_rad = np.deg2rad(self.attitude[2])
            speed_estimate = abs(vz) * 2.0  # Rough estimate
            self.velocity = np.array([
                speed_estimate * np.cos(heading_rad),
                speed_estimate * np.sin(heading_rad),
                vz
            ], dtype=np.float32)
            self.altitude_rate = vz
        else:
            # Random initial state
            self.velocity = np.array([
                np.random.uniform(-5.0, 5.0),
                np.random.uniform(-5.0, 5.0),
                np.random.uniform(-2.0, 2.0)
            ], dtype=np.float32)
            self.attitude = np.array([
                np.random.uniform(-5.0, 5.0),
                np.random.uniform(-5.0, 5.0),
                np.random.uniform(0.0, 360.0)
            ], dtype=np.float32)
            self.altitude_rate = self.velocity[2]
        
        # SABİT HEDEF POZİSYONU - MERKEZE YAKIN, BAŞLANGIÇTAN UZAK
        # Her episode'da aynı yerde hedef
        # Başlangıçtan 200m uzaklıkta, kuzey yönünde
        target_distance = 200.0  # Sabit mesafe
        target_angle = 0.0  # Kuzey yönü (sabit)
        
        target_x = self.position[0] + target_distance * np.cos(target_angle)  # 200m kuzey
        target_y = self.position[1] + target_distance * np.sin(target_angle)  # 0m (aynı y)
        target_z = 100.0  # Aynı irtifa (sabit)
        
        self.target_position = np.array([target_x, target_y, target_z], dtype=np.float32)
        self.last_distance_to_target = np.linalg.norm(self.position - self.target_position)
        self.step_count = 0
        self.episode_rewards = []
        self.prev_roll = self.attitude[0]
        self.prev_pitch = self.attitude[1]
        
        return self._get_state(), {
            'position': self.position.copy(),
            'target': self.target_position.copy(),
            'distance_to_target': self.last_distance_to_target
        }
    
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
        
        # Check collisions
        collision = False
        boundary_penalty = 0.0
        
        world_half_x = self.world_size[0] / 2
        world_half_y = self.world_size[1] / 2
        world_max_z = self.world_size[2]
        
        # Boundary warning - daha erken uyarı
        boundary_warning = False
        if (abs(self.position[0]) > world_half_x * 0.7 or 
            abs(self.position[1]) > world_half_y * 0.7 or
            self.position[2] < 30.0 or self.position[2] > world_max_z * 0.85):
            boundary_warning = True
        
        # Boundary collision
        if (self.position[0] < -world_half_x or self.position[0] > world_half_x or
            self.position[1] < -world_half_y or self.position[1] > world_half_y or
            self.position[2] < 2.0 or self.position[2] > world_max_z):
            collision = True
            boundary_penalty = -100.0  # DAHA DA AZALTILDI: -200 → -100
        
        # Calculate reward
        reward = 0.0
        
        # Distance to target - her durumda hesapla (debug için gerekli)
        distance_to_target = np.linalg.norm(self.position - self.target_position)
        goal_reached = False  # Her durumda başlat
        
        if collision:
            reward += boundary_penalty
        else:
            # ALTITUDE KORUMA
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
            
            if self.position[2] < 50.0:
                reward += -10.0 * (1.0 - self.position[2] / 50.0)
            
            # BOUNDARY'DEN UZAK DURMA BONUSU
            center_distance = np.sqrt(
                (self.position[0] / world_half_x) ** 2 +
                (self.position[1] / world_half_y) ** 2
            )
            if center_distance < 0.6:  # Merkeze yakınsa
                boundary_safety_bonus = 1.0 * (1.0 - center_distance / 0.6)  # Artırıldı: 0.5 → 1.0
                reward += boundary_safety_bonus
            
            if boundary_warning:
                reward += -1.0  # Artırıldı: -0.5 → -1.0
            
            # HEDEFE ULAŞMA - ÇOK GÜÇLENDİRİLDİ
            # distance_to_target zaten yukarıda hesaplandı
            progress = self.last_distance_to_target - distance_to_target
            
            # İlerleme ödülü - final approach için güçlendirildi
            if progress > 0:
                distance_factor = max(0.6, 1.0 - distance_to_target / 1000.0)  # 500 → 1000
                
                # Final approach için progress reward'u DAHA DA güçlendir
                if distance_to_target < 15.0:  # Son 15m - ÇOK GÜÇLÜ
                    progress_multiplier = 80.0 * (1.0 + 2.5 * (15.0 - distance_to_target) / 15.0)  # 60x → 200x (max)
                elif distance_to_target < 25.0:  # Son 25m - GÜÇLÜ
                    progress_multiplier = 70.0 * (1.0 + 1.5 * (25.0 - distance_to_target) / 25.0)  # 60x → 140x (max)
                elif distance_to_target < 50.0:  # YENİ: 25-50m arası
                    progress_multiplier = 65.0 * (1.0 + 0.5 * (50.0 - distance_to_target) / 25.0)  # 60x → 80x (max)
                else:
                    progress_multiplier = 60.0
                
                progress_reward = progress * self.progress_reward * progress_multiplier * distance_factor
                reward += progress_reward
            
            # Büyük ilerleme bonusu (10m+ ilerleme)
            if progress > 10.0:
                reward += 20.0  # Büyük ilerleme bonusu
            
            self.last_distance_to_target = distance_to_target
            
            # YAW/HEADING ALIGNMENT REWARD - YENİ EKLENECEK
            # Hedefe doğru yönelme ödülü (daha düz uçuş için)
            target_direction = self.target_position - self.position
            target_direction[2] = 0  # Sadece yatay düzlemde
            target_direction_norm = np.linalg.norm(target_direction)
            if target_direction_norm > 0.1:
                target_direction = target_direction / target_direction_norm
                
                # Velocity yönü
                velocity_horizontal = self.velocity.copy()
                velocity_horizontal[2] = 0  # Sadece yatay düzlemde
                velocity_norm = np.linalg.norm(velocity_horizontal)
                
                if velocity_norm > 0.1:
                    velocity_direction = velocity_horizontal / velocity_norm
                    
                    # Velocity ve target direction arasındaki açı
                    alignment_dot = np.clip(np.dot(velocity_direction, target_direction), -1.0, 1.0)
                    alignment_angle = np.arccos(alignment_dot) * 180.0 / np.pi  # Derece
                    
                    # Yaw açısı (heading) ile target direction arasındaki açı
                    yaw_rad = np.deg2rad(self.attitude[2])
                    heading_direction = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
                    yaw_alignment_dot = np.clip(np.dot(heading_direction[:2], target_direction[:2]), -1.0, 1.0)
                    yaw_alignment_angle = np.arccos(yaw_alignment_dot) * 180.0 / np.pi  # Derece
                    
                    # Alignment reward - final approach'ta daha güçlü
                    if distance_to_target < 50.0:  # Final approach'ta çok önemli
                        alignment_reward = 10.0 * max(0.0, 1.0 - alignment_angle / 45.0)  # 45° içinde ödül
                        yaw_alignment_reward = 8.0 * max(0.0, 1.0 - yaw_alignment_angle / 45.0)  # 45° içinde ödül
                    elif distance_to_target < 100.0:
                        alignment_reward = 5.0 * max(0.0, 1.0 - alignment_angle / 60.0)  # 60° içinde ödül
                        yaw_alignment_reward = 3.0 * max(0.0, 1.0 - yaw_alignment_angle / 60.0)
                    else:
                        alignment_reward = 2.0 * max(0.0, 1.0 - alignment_angle / 90.0)  # 90° içinde ödül
                        yaw_alignment_reward = 1.0 * max(0.0, 1.0 - yaw_alignment_angle / 90.0)
                    
                    reward += alignment_reward
                    reward += yaw_alignment_reward
            
            # Goal reward - DAHA DA GÜÇLENDİRİLDİ
            if distance_to_target < 15.0:  # Hedefe ulaşıldı!
                reward += self.goal_reward * 6.0  # 5.0 → 6.0 (12000 puan!)
                proximity_bonus = 1500.0 * (1.0 - distance_to_target / 15.0)  # 1000 → 1500
                reward += proximity_bonus
                # Final approach'ta ekstra bonus
                final_approach_super_bonus = 2000.0 * (1.0 - distance_to_target / 15.0)  # YENİ
                reward += final_approach_super_bonus
                goal_reached = True  # Hedefe ulaşıldı işareti
                print(f"[GOAL REACHED] Mesafe: {distance_to_target:.2f}m - Episode bitiyor!")
            elif distance_to_target < 20.0:  # Final approach (15-20m arası) - GÜÇLENDİRİLDİ
                final_approach_bonus = 1200.0 * (1.0 - (distance_to_target - 15.0) / 5.0)  # 800 → 1200
                reward += final_approach_bonus
                proximity_bonus = 600.0 * (1.0 - distance_to_target / 20.0)  # 400 → 600
                reward += proximity_bonus
                # Ekstra yakınlık bonusu
                extra_proximity = 500.0 * (1.0 - (distance_to_target - 15.0) / 5.0)  # YENİ
                reward += extra_proximity
            elif distance_to_target < 25.0:  # Yakın approach (20-25m arası) - GÜÇLENDİRİLDİ
                proximity_bonus = 400.0 * (1.0 - (distance_to_target - 20.0) / 5.0)  # 200 → 400
                reward += proximity_bonus
                # Orta yakınlık bonusu
                medium_proximity = 300.0 * (1.0 - (distance_to_target - 20.0) / 5.0)  # YENİ
                reward += medium_proximity
            elif distance_to_target < 30.0:  # YENİ: 25-30m arası
                proximity_bonus = 200.0 * (1.0 - (distance_to_target - 25.0) / 5.0)
                reward += proximity_bonus
            elif distance_to_target < 50.0:  # Yakın - ARTIRILDI
                proximity_bonus = 100.0 * (1.0 - distance_to_target / 50.0)  # 50 → 100
                reward += proximity_bonus
            elif distance_to_target < 100.0:  # Orta mesafe - ARTIRILDI
                proximity_bonus = 50.0 * (1.0 - distance_to_target / 100.0)  # 20 → 50
                reward += proximity_bonus
            elif distance_to_target < 150.0:  # YENİ: Orta-yakın mesafe
                proximity_bonus = 30.0 * (1.0 - (distance_to_target - 100.0) / 50.0)
                reward += proximity_bonus
            
            # Survival bonus
            reward += 0.4  # Artırıldı: 0.3 → 0.4
        
        self.episode_rewards.append(reward)
        self.step_count += 1
        
        # Termination conditions
        terminated = collision or goal_reached  # Hedefe ulaşıldığında episode bitir
        truncated = (self.step_count >= self.max_episode_steps)
        
        # Debug: Hedefe yakınsa mesafeyi göster
        if distance_to_target < 25.0 and not goal_reached:
            if self.step_count % 10 == 0:  # Her 10 adımda bir göster
                print(f"[DEBUG] Adım {self.step_count}: Hedefe mesafe = {distance_to_target:.2f}m (Eşik: 15m)")
        
        info = {
            'position': self.position.copy(),
            'target': self.target_position.copy(),
            'distance_to_target': self.last_distance_to_target,
            'collision': collision,
            'goal_reached': goal_reached,  # Hedefe ulaşıldı mı?
            'altitude': self.position[2]
        }
        
        return self._get_state(), reward, terminated, truncated, info


def make_env_3d(env_config, rank=0):
    """Create and wrap 3D environment with optimizations."""
    def _init():
        env = OptimizedTargetReachingEnv(**env_config)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def train_stage1_optimized(
    total_timesteps=2500000,  # UZUN EĞİTİM: 2.5M steps (daha iyi öğrenme için)
    learning_rate=2e-4,
    model_name=None,
    log_dir="./logs_3d/",
    save_dir="./models_3d/",
    resume_from_checkpoint=None  # Checkpoint'ten devam et
):
    """
    Aşama 1 eğitimi - Tüm optimizasyonlar uygulandı.
    
    Args:
        resume_from_checkpoint: Checkpoint dosya yolu (örn: './models_3d/stage1_optimized_20260104_043539_checkpoints/stage1_optimized_20260104_043539_180000_steps.zip')
    """
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint'ten devam ediliyorsa model adını checkpoint'ten al
    if resume_from_checkpoint:
        if not os.path.exists(resume_from_checkpoint):
            print(f"[HATA] Checkpoint bulunamadi: {resume_from_checkpoint}")
            return None, None
        
        # Checkpoint dosya adından model adını çıkar
        checkpoint_name = os.path.basename(resume_from_checkpoint)
        # Örnek: stage1_optimized_20260104_043539_180000_steps.zip -> stage1_optimized_20260104_043539
        model_name = '_'.join(checkpoint_name.split('_')[:-2])  # Son 2 parçayı çıkar (_180000_steps)
        print(f"[BILGI] Checkpoint'ten devam ediliyor: {resume_from_checkpoint}")
        print(f"[BILGI] Model adi: {model_name}")
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"stage1_optimized_{timestamp}"
    
    print("=" * 70)
    print("ASAMA 1 EGITIMI - TUM OPTIMIZASYONLAR UYGULANDI")
    print("=" * 70)
    print(f"Model Adi: {model_name}")
    print(f"Toplam Timesteps: {total_timesteps:,}")
    print(f"\n[UYGULANAN OPTIMIZASYONLAR]")
    print(f"  1. Dunya boyutu artirildi: 500x500x200 -> 1000x1000x300")
    print(f"  2. SABIT baslangic pozisyonu: (0, 0, 100) - tam merkez")
    print(f"  3. SABIT hedef pozisyonu: (200, 0, 100) - 200m kuzey")
    print(f"  4. Hiz limiti azaltildi: 25 m/s -> 15 m/s")
    print(f"  5. Boundary penalty azaltildi: -200 -> -100")
    print(f"  6. Hedefe ulasma reward cok guclendirildi (6000 puan!)")
    print(f"  7. Ilerleme reward artirildi (60x multiplier)")
    print(f"  8. Buyuk ilerleme bonusu eklendi (10m+ ilerleme)")
    print(f"  9. Episode uzunlugu artirildi: 2000 -> 5000 adim")
    print(f"  10. Survival bonus artirildi: 0.3 -> 0.4")
    print(f"  11. Hedefe ulasma esigi 15m (episode bitiyor)")
    print(f"  12. ASAMA 1: Son 10-15m icin guclendirilmis reward'lar eklendi")
    print(f"  13. ASAMA 1: Proximity bonus'lar artirildi (10m: 1000, 15m: 500)")
    print(f"  14. ASAMA 1: Progress reward son 20m'de 2x artirildi (60x -> 120x)")
    print(f"  15. YENİ: Yaw/Heading alignment reward eklendi (daha düz uçuş)")
    print(f"  16. YENİ: Final approach reward'ları güçlendirildi (goal: 12K, proximity: 1.5K)")
    print(f"  17. YENİ: Proximity bonus'lar artırıldı (50m: 100, 100m: 50, 150m: 30)")
    print(f"  18. YENİ: Progress multiplier güçlendirildi (15m: 200x, 25m: 140x, 50m: 80x)")
    print(f"  19. YENİ: Eğitim süresi artırıldı (1.5M -> 2.5M steps)")
    print("=" * 70)
    
    # OPTIMIZE EDILMIS ENVIRONMENT CONFIG
    env_config = {
        'world_size': (1000.0, 1000.0, 300.0),  # ARTIRILDI: 500x500x200 -> 1000x1000x300
        'num_obstacles': 0,  # ENGEL YOK!
        'max_episode_steps': 5000,  # ARTIRILDI: 2000 -> 5000
        'dt': 0.1,
        'max_speed': 15.0,  # AZALTILDI: 25.0 -> 15.0
        'collision_penalty': -100.0,  # AZALTILDI: -200 -> -100
        'obstacle_penalty': 0.0,
        'goal_reward': 2000.0,  # Base (final'de 3x = 6000)
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
    
    # Checkpoint'ten devam ediliyorsa yükle, yoksa yeni model oluştur
    if resume_from_checkpoint:
        print(f"[YUKLENIYOR] Checkpoint yukleniyor: {resume_from_checkpoint}")
        model = PPO.load(resume_from_checkpoint, env=env)
        # Learning rate'i güncelle (eğer değiştirildiyse)
        if model.learning_rate != learning_rate:
            model.learning_rate = learning_rate
            print(f"[GUNCELLENDI] Learning rate: {learning_rate}")
        print(f"[OK] Checkpoint yuklendi! Mevcut timesteps: {model.num_timesteps:,}")
    else:
        model = PPO('MlpPolicy', env, **ppo_config)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, f"{model_name}_best"),
        log_path=os.path.join(log_dir, f"{model_name}_eval"),
        eval_freq=10000,  # Her 10K steps'te evaluation (kısa eğitim için)
        n_eval_episodes=5,  # Kısa eğitim için azaltıldı
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Her 10K steps'te checkpoint (kısa eğitim için)
        save_path=os.path.join(save_dir, f"{model_name}_checkpoints"),
        name_prefix=model_name
    )
    
    print(f"\n[BASLIYOR] Egitim basliyor...")
    print(f"[BILGI] TensorBoard: tensorboard --logdir {log_dir}")
    
    # Checkpoint'ten devam ediliyorsa reset_num_timesteps=False (timesteps'i koru)
    reset_timesteps = not resume_from_checkpoint
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=reset_timesteps  # Checkpoint'ten devam ediyorsa False
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
    parser = argparse.ArgumentParser(description='Aşama 1 eğitimi - Tüm optimizasyonlar')
    
    parser.add_argument('--total_timesteps', type=int, default=2500000,
                       help='Toplam eğitim adımı (default: 2.5M - uzun eğitim için)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Öğrenme oranı')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model adı')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Checkpoint dosya yolu (örn: ./models_3d/stage1_optimized_20260104_043539_checkpoints/stage1_optimized_20260104_043539_180000_steps.zip)')
    
    args = parser.parse_args()
    
    train_stage1_optimized(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

