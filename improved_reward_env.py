"""
İyileştirilmiş Reward Fonksiyonu ile Environment Wrapper
Başarı oranını artırmak için optimize edilmiş reward shaping
"""

import numpy as np
from gymnasium import Wrapper
from typing import Tuple, Dict


class ImprovedRewardWrapper(Wrapper):
    """
    Reward fonksiyonunu optimize eden wrapper.
    Daha dengeli ve hedefe odaklı reward shaping.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Reward parametreleri - optimize edilmiş
        self.goal_reward = 2000.0  # Hedefe ulaşma için çok büyük ödül
        self.progress_multiplier = 30.0  # İlerleme için güçlü ödül
        self.obstacle_safe_distance = 50.0  # Güvenli mesafe
        self.obstacle_warning_distance = 100.0  # Uyarı mesafesi
        
        # Tracking
        self.last_distance_to_target = None
        self.last_obstacle_distance = None
        
    def reset(self, **kwargs):
        """Reset environment and tracking variables."""
        obs, info = self.env.reset(**kwargs)
        self.last_distance_to_target = info.get('distance_to_target', float('inf'))
        # Reset'te min_obstacle_distance olmayabilir, environment'dan al
        if 'min_obstacle_distance' in info:
            self.last_obstacle_distance = info['min_obstacle_distance']
        else:
            # Environment'dan direkt al
            if hasattr(self.env, '_get_min_obstacle_distance'):
                self.last_obstacle_distance = self.env._get_min_obstacle_distance(info.get('position', np.array([0, 0, 0])))
            else:
                self.last_obstacle_distance = float('inf')
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with improved reward function."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Yeni reward hesapla
        improved_reward = self._compute_improved_reward(
            obs, reward, terminated, truncated, info
        )
        
        return obs, improved_reward, terminated, truncated, info
    
    def _compute_improved_reward(
        self, 
        obs: np.ndarray, 
        original_reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict
    ) -> float:
        """
        İyileştirilmiş reward fonksiyonu.
        
        Strateji:
        1. Hedefe ulaşma için çok büyük ödül
        2. İlerleme için güçlü pozitif ödül
        3. Engelden kaçınma için dengeli penalty
        4. Çarpışma için büyük ama aşırı olmayan penalty
        """
        
        new_reward = 0.0
        
        # 1. ÇARPIŞMA KONTROLÜ
        if terminated and info.get('collision', False):
            # Çarpışma penalty - yeterince büyük ama çok fazla değil
            new_reward += -1000.0
            return new_reward
        
        # 2. HEDEFE ULAŞMA (en önemli!)
        distance_to_target = info['distance_to_target']
        
        # Hedefe çok yakınsa (10m içinde) - ÇOK BÜYÜK ÖDÜL
        if distance_to_target < 10.0:
            new_reward += self.goal_reward  # 2000 puan!
            # Ekstra bonus: hedefe çok yakınsa
            proximity_bonus = 100.0 * (1.0 - distance_to_target / 10.0)
            new_reward += proximity_bonus
        
        # Hedefe yakınsa (50m içinde) - güçlü ödül
        elif distance_to_target < 50.0:
            proximity_reward = 50.0 * (1.0 - distance_to_target / 50.0)
            new_reward += proximity_reward
        
        # Hedefe orta mesafede (100m içinde) - orta ödül
        elif distance_to_target < 100.0:
            proximity_reward = 20.0 * (1.0 - distance_to_target / 100.0)
            new_reward += proximity_reward
        
        # 3. İLERLEME ÖDÜLÜ (hedefe yaklaşma)
        if self.last_distance_to_target is not None:
            progress = self.last_distance_to_target - distance_to_target
            
            if progress > 0:  # İlerleme varsa
                # Mesafeye göre scale et (yakınsa daha fazla ödül)
                distance_factor = max(0.3, 1.0 - distance_to_target / 500.0)
                progress_reward = progress * self.progress_multiplier * distance_factor
                new_reward += progress_reward
            elif progress < -5.0:  # Geri gidiyorsa küçük penalty
                new_reward += progress * 2.0  # Geri gitme penalty
        
        self.last_distance_to_target = distance_to_target
        
        # 4. ENGELDEN KAÇINMA
        min_obstacle_dist = info.get('min_obstacle_distance', float('inf'))
        
        if min_obstacle_dist < float('inf'):
            # Çok yakınsa (20m içinde) - büyük penalty
            if min_obstacle_dist < 20.0:
                penalty = -200.0 * (1.0 - min_obstacle_dist / 20.0)
                new_reward += penalty
            
            # Yakınsa (50m içinde) - orta penalty
            elif min_obstacle_dist < 50.0:
                penalty = -50.0 * (1.0 - min_obstacle_dist / 50.0)
                new_reward += penalty
            
            # Uyarı mesafesinde (100m içinde) - küçük penalty
            elif min_obstacle_dist < 100.0:
                penalty = -10.0 * (1.0 - min_obstacle_dist / 100.0)
                new_reward += penalty
            
            # Engelden uzaklaşırsa bonus
            if self.last_obstacle_distance is not None:
                escape = min_obstacle_dist - self.last_obstacle_distance
                if escape > 0 and min_obstacle_dist < 100.0:
                    # Uzaklaşma ödülü
                    new_reward += 10.0 * escape
            
            # Güvenli mesafede kalırsa küçük bonus
            if min_obstacle_dist > 50.0 and min_obstacle_dist < 150.0:
                new_reward += 0.5  # Güvenli mesafe bonusu
        
        self.last_obstacle_distance = min_obstacle_dist
        
        # 5. STABİLİTE ÖDÜLÜ (küçük ama önemli)
        # Çok agresif değil - sadece aşırı durumlarda penalty
        position = info.get('position', np.array([0, 0, 0]))
        velocity = obs[3:6] if len(obs) >= 6 else np.array([0, 0, 0])
        
        speed = np.linalg.norm(velocity)
        if speed > 30.0:  # Çok hızlıysa küçük penalty
            new_reward += -0.1 * (speed - 30.0)
        
        # 6. SURVIVAL BONUS (hayatta kalma ödülü)
        # Her adım hayatta kalırsa küçük ödül (exploration teşviki)
        if not terminated:
            new_reward += 0.1  # Survival bonus
        
        return new_reward

