"""
Waypoint-Based Navigation Environment
Gelecekteki waypoint'leri hedefe doğru yönlendirir.
Eğer 3. waypoint hedeften geçiyorsa manevra yapıp rotayı düzenler.
"""

import numpy as np
from gymnasium import Wrapper
from typing import Tuple, Dict, List


class WaypointNavigationWrapper(Wrapper):
    """
    Waypoint-based navigation wrapper.
    Gelecekteki waypoint'leri hedefe doğru yönlendirir.
    """
    
    def __init__(self, env, num_waypoints=3, waypoint_distance=50.0):
        """
        Args:
            env: Base environment
            num_waypoints: Kaç waypoint önceden hesaplanacak (default: 3)
            waypoint_distance: Waypoint'ler arası mesafe (metre)
        """
        super().__init__(env)
        self.num_waypoints = num_waypoints
        self.waypoint_distance = waypoint_distance
        self.waypoints = []
        
    def reset(self, **kwargs):
        """Reset environment and calculate waypoints."""
        obs, info = self.env.reset(**kwargs)
        self._update_waypoints(info)
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with waypoint guidance."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Waypoint'leri güncelle
        self._update_waypoints(info)
        
        # Waypoint bilgisini info'ya ekle
        info['waypoints'] = self.waypoints.copy()
        info['next_waypoint'] = self.waypoints[0] if self.waypoints else None
        info['waypoint_direction'] = self._get_waypoint_direction(info)
        
        # Waypoint'e yönelme bonusu
        waypoint_bonus = self._calculate_waypoint_bonus(info)
        reward += waypoint_bonus
        
        return obs, reward, terminated, truncated, info
    
    def _update_waypoints(self, info: Dict):
        """Hedefe doğru waypoint'leri hesapla."""
        position = info.get('position', np.array([0, 0, 0]))
        target = info.get('target', np.array([0, 0, 0]))
        velocity = info.get('velocity', np.array([0, 0, 0]))
        
        # Hedefe doğru vektör
        to_target = target - position
        distance_to_target = np.linalg.norm(to_target)
        
        if distance_to_target < 0.1:
            # Hedefe çok yakınsa waypoint yok
            self.waypoints = []
            return
        
        # Hedefe doğru birim vektör
        direction = to_target / distance_to_target
        
        # Mevcut hız yönü
        speed = np.linalg.norm(velocity)
        if speed > 0.1:
            velocity_direction = velocity / speed
        else:
            velocity_direction = direction
        
        # Waypoint'leri hesapla
        self.waypoints = []
        current_pos = position.copy()
        
        for i in range(self.num_waypoints):
            # Bir sonraki waypoint pozisyonu
            # Mevcut hız yönüne doğru ama hedefe de yakın
            waypoint_direction = 0.7 * direction + 0.3 * velocity_direction
            waypoint_direction = waypoint_direction / np.linalg.norm(waypoint_direction)
            
            next_waypoint = current_pos + waypoint_direction * self.waypoint_distance
            
            # Eğer waypoint hedefi geçiyorsa, hedefe doğru düzelt
            to_waypoint_from_target = next_waypoint - target
            if np.dot(to_waypoint_from_target, direction) < 0:
                # Waypoint hedefin arkasında, hedefe doğru düzelt
                next_waypoint = target - direction * (self.waypoint_distance * (self.num_waypoints - i))
            
            self.waypoints.append(next_waypoint)
            current_pos = next_waypoint.copy()
    
    def _get_waypoint_direction(self, info: Dict) -> np.ndarray:
        """Bir sonraki waypoint'e doğru yön vektörü."""
        if not self.waypoints:
            # Waypoint yoksa hedefe doğru
            position = info.get('position', np.array([0, 0, 0]))
            target = info.get('target', np.array([0, 0, 0]))
            direction = target - position
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 0.1 else np.array([1, 0, 0])
        
        position = info.get('position', np.array([0, 0, 0]))
        next_waypoint = self.waypoints[0]
        direction = next_waypoint - position
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 0.1 else np.array([1, 0, 0])
    
    def _calculate_waypoint_bonus(self, info: Dict) -> float:
        """Waypoint'e yönelme için bonus reward."""
        if not self.waypoints:
            return 0.0
        
        position = info.get('position', np.array([0, 0, 0]))
        velocity = info.get('velocity', np.array([0, 0, 0]))
        next_waypoint = self.waypoints[0]
        
        # Waypoint'e doğru vektör
        to_waypoint = next_waypoint - position
        distance_to_waypoint = np.linalg.norm(to_waypoint)
        
        if distance_to_waypoint < 0.1:
            # Waypoint'e ulaşıldı - bonus!
            return 10.0
        
        # Hız yönü waypoint'e doğru mu?
        speed = np.linalg.norm(velocity)
        if speed > 0.1:
            velocity_direction = velocity / speed
            waypoint_direction = to_waypoint / distance_to_waypoint
            alignment = np.dot(velocity_direction, waypoint_direction)
            
            # Yönlendirme bonusu (0.0 - 1.0 arası)
            if alignment > 0.7:  # İyi hizalama
                return 0.5 * alignment
            elif alignment < -0.5:  # Yanlış yön
                return -0.2
        
        return 0.0
    
    def render(self, mode='human'):
        """Render with waypoints."""
        # Base render
        if hasattr(self.env, 'render'):
            self.env.render(mode=mode)
        
        # Waypoint'leri çiz (eğer matplotlib kullanılıyorsa)
        if mode == 'human' and hasattr(self.env, 'ax') and self.env.ax is not None:
            ax = self.env.ax
            
            # Waypoint'leri çiz
            for i, waypoint in enumerate(self.waypoints):
                color = ['yellow', 'orange', 'red'][i % 3]
                ax.scatter(
                    [waypoint[0]], [waypoint[1]], [waypoint[2]],
                    c=color, marker='o', s=100, alpha=0.7,
                    label=f'Waypoint {i+1}' if i == 0 else ''
                )
            
            # Waypoint'ler arası çizgiler
            if len(self.waypoints) > 1:
                waypoint_array = np.array(self.waypoints)
                ax.plot(
                    waypoint_array[:, 0],
                    waypoint_array[:, 1],
                    waypoint_array[:, 2],
                    'y--', alpha=0.5, linewidth=1
                )

