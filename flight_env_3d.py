"""
3D Flight Control Environment with Obstacles
Helicopter must navigate in 3D space while avoiding obstacles.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import os
from state_extractor import StateExtractor


class Obstacle3D:
    """3D obstacle (cylinder or sphere) - can be moving."""
    
    def __init__(
        self,
        position: np.ndarray,  # [x, y, z]
        radius: float,
        height: Optional[float] = None,  # None = sphere, value = cylinder
        obstacle_type: str = 'cylinder',
        velocity: Optional[np.ndarray] = None,  # [vx, vy, vz] for moving obstacles
        movement_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # [min_pos, max_pos]
    ):
        self.position = np.array(position, dtype=np.float32)
        self.radius = radius
        self.height = height
        self.type = obstacle_type
        self.velocity = np.array(velocity, dtype=np.float32) if velocity is not None else None
        self.movement_bounds = movement_bounds
        self.initial_position = self.position.copy()
    
    def update(self, dt: float):
        """Update obstacle position if it's moving."""
        if self.velocity is not None:
            self.position += self.velocity * dt
            
            # Bounce off bounds if specified
            if self.movement_bounds is not None:
                min_pos, max_pos = self.movement_bounds
                for i in range(3):
                    if self.position[i] < min_pos[i]:
                        self.position[i] = min_pos[i]
                        self.velocity[i] = -self.velocity[i]
                    elif self.position[i] > max_pos[i]:
                        self.position[i] = max_pos[i]
                        self.velocity[i] = -self.velocity[i]
    
    def check_collision(self, position: np.ndarray, safety_radius: float = 1.0) -> bool:
        """Check if position collides with obstacle."""
        if self.type == 'sphere':
            distance = np.linalg.norm(position - self.position)
            return distance < (self.radius + safety_radius)
        elif self.type == 'cylinder':
            # Check horizontal distance
            horizontal_dist = np.linalg.norm(position[:2] - self.position[:2])
            if horizontal_dist > (self.radius + safety_radius):
                return False
            # Check vertical bounds
            z_min = self.position[2]
            z_max = self.position[2] + (self.height if self.height else float('inf'))
            return z_min - safety_radius <= position[2] <= z_max + safety_radius
        return False
    
    def distance_to_obstacle(self, position: np.ndarray) -> float:
        """Calculate minimum distance to obstacle surface."""
        if self.type == 'sphere':
            distance = np.linalg.norm(position - self.position)
            return max(0, distance - self.radius)
        elif self.type == 'cylinder':
            horizontal_dist = np.linalg.norm(position[:2] - self.position[:2])
            radial_dist = max(0, horizontal_dist - self.radius)
            
            z_min = self.position[2]
            z_max = self.position[2] + (self.height if self.height else float('inf'))
            
            if position[2] < z_min:
                vertical_dist = z_min - position[2]
            elif position[2] > z_max:
                vertical_dist = position[2] - z_max
            else:
                vertical_dist = 0
            
            return np.sqrt(radial_dist**2 + vertical_dist**2)
        return float('inf')


class FlightControlEnv3D(gym.Env):
    """
    3D Flight Control Environment with Obstacles.
    
    State Space (11D):
    - position: [x, y, z] (meters)
    - velocity: [vx, vy, vz] (m/s)
    - attitude: [roll, pitch, yaw] (degrees)
    - altitude_rate: (m/s)
    
    Action Space (4D):
    - roll_command: [-1, 1]
    - pitch_command: [-1, 1]
    - yaw_command: [-1, 1]
    - throttle_command: [-1, 1]
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        world_size: Tuple[float, float, float] = (500.0, 500.0, 200.0),  # [x, y, z] in meters
        num_obstacles: int = 5,
        obstacle_radius_range: Tuple[float, float] = (10.0, 30.0),
        obstacle_height_range: Tuple[float, float] = (50.0, 150.0),
        target_position: Optional[np.ndarray] = None,
        initial_position_range: Tuple[float, float] = (50.0, 150.0),
        max_speed: float = 50.0,  # m/s
        max_episode_steps: int = 2000,
        dt: float = 0.1,
        collision_penalty: float = -100.0,
        obstacle_penalty: float = -50.0,
        goal_reward: float = 100.0,
        progress_reward: float = 0.1,
        render_mode: Optional[str] = None,
        use_log_data: bool = False,  # Use FlightGear log data for initialization
        log_data_path: str = "fg_log2.csv",  # Path to log data
        moving_obstacles: bool = True,  # Enable moving obstacles
        obstacle_speed: float = 5.0,  # Speed of moving obstacles (m/s)
        target_behind_obstacle: bool = True  # Place target behind an obstacle
    ):
        """
        Initialize 3D flight control environment.
        
        Args:
            world_size: Size of 3D world [x, y, z] in meters
            num_obstacles: Number of obstacles
            obstacle_radius_range: Range for obstacle radius
            obstacle_height_range: Range for obstacle height
            target_position: Target position (None = random)
            initial_position_range: Range for initial z position
            max_speed: Maximum speed (m/s)
            max_episode_steps: Maximum steps per episode
            dt: Time step (seconds)
            collision_penalty: Penalty for collision
            obstacle_penalty: Penalty for getting too close to obstacle
            goal_reward: Reward for reaching target
            progress_reward: Reward per step for making progress
            render_mode: Rendering mode
        """
        super(FlightControlEnv3D, self).__init__()
        
        self.world_size = np.array(world_size, dtype=np.float32)
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.obstacle_height_range = obstacle_height_range
        self.target_position = target_position
        self.initial_position_range = initial_position_range
        self.max_speed = max_speed
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.collision_penalty = collision_penalty
        self.obstacle_penalty = obstacle_penalty
        self.goal_reward = goal_reward
        self.progress_reward = progress_reward
        self.render_mode = render_mode
        self.moving_obstacles = moving_obstacles
        self.obstacle_speed = obstacle_speed
        self.target_behind_obstacle = target_behind_obstacle
        
        # State space: [x, y, z, vx, vy, vz, roll, pitch, yaw, altitude_rate]
        self.observation_space = spaces.Box(
            low=np.array([
                -world_size[0]/2, -world_size[1]/2, 0.0,  # position
                -max_speed, -max_speed, -max_speed,  # velocity
                -90.0, -90.0, 0.0,  # attitude (roll, pitch, yaw)
                -50.0  # altitude_rate
            ], dtype=np.float32),
            high=np.array([
                world_size[0]/2, world_size[1]/2, world_size[2],  # position
                max_speed, max_speed, max_speed,  # velocity
                90.0, 90.0, 360.0,  # attitude
                50.0  # altitude_rate
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [roll, pitch, yaw, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # State variables
        self.position = None
        self.velocity = None
        self.attitude = None  # [roll, pitch, yaw]
        self.altitude_rate = None
        self.obstacles = []
        self.step_count = 0
        self.episode_rewards = []
        self.last_distance_to_target = None
        
        # Log data for realistic initialization
        self.use_log_data = use_log_data
        self.log_states = None
        if use_log_data and os.path.exists(log_data_path):
            try:
                extractor = StateExtractor(collision_threshold=2.0)
                df = extractor.load_data(log_data_path)
                state_df = extractor.extract_states(df)
                # Store state vectors: [altitude_agl, roll, pitch, heading, altitude_rate]
                self.log_states = state_df[['altitude_agl', 'roll', 'pitch', 'heading', 'altitude_rate']].values
                # Also store longitude for x position approximation
                self.log_longitude = df['Longitude'].values
                print(f"[OK] Loaded {len(self.log_states)} states from {log_data_path}")
            except Exception as e:
                print(f"[WARNING] Could not load log data: {e}. Using random initialization.")
                self.use_log_data = False
        elif use_log_data:
            print(f"[WARNING] Log data file not found: {log_data_path}. Using random initialization.")
            self.use_log_data = False
        
        # For rendering
        self.fig = None
        self.ax = None
        
    def _generate_obstacles(self, seed: Optional[int] = None):
        """Generate single fixed obstacle (for target-behind scenario)."""
        if seed is not None:
            np.random.seed(seed)
        
        self.obstacles = []
        self.main_obstacle = None
        
        # Generate single obstacle in the middle of the path
        # Position it between typical start and target positions
        x = np.random.uniform(-50, 50)  # Center area
        y = np.random.uniform(-50, 50)  # Center area
        z = np.random.uniform(50, 150)  # Mid altitude
        
        # Make obstacle substantial but not too large
        radius = np.random.uniform(20.0, 40.0)
        height = np.random.uniform(80.0, 120.0)
        
        # Fixed obstacle (no velocity, not moving)
        obstacle = Obstacle3D(
            position=np.array([x, y, z]),
            radius=radius,
            height=height,
            obstacle_type='cylinder',
            velocity=None,  # Fixed obstacle
            movement_bounds=None
        )
        self.obstacles.append(obstacle)
        self.main_obstacle = obstacle  # This is the main obstacle (target will be behind it)
    
    def _check_collisions(self, position: np.ndarray) -> Tuple[bool, Optional[Obstacle3D]]:
        """Check for collisions with obstacles or boundaries."""
        # Check world boundaries
        if (position[0] < -self.world_size[0]/2 or position[0] > self.world_size[0]/2 or
            position[1] < -self.world_size[1]/2 or position[1] > self.world_size[1]/2 or
            position[2] < 2.0 or position[2] > self.world_size[2]):
            return True, None
        
        # Check obstacle collisions
        for obstacle in self.obstacles:
            if obstacle.check_collision(position, safety_radius=2.0):
                return True, obstacle
        
        return False, None
    
    def _get_min_obstacle_distance(self, position: np.ndarray) -> float:
        """Get minimum distance to any obstacle."""
        if not self.obstacles:
            return float('inf')
        return min([obs.distance_to_obstacle(position) for obs in self.obstacles])
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate obstacles
        self._generate_obstacles(seed)
        
        # Initial position and state from log data or random
        selected_log_idx = None
        if self.use_log_data and self.log_states is not None and len(self.log_states) > 0:
            # Try to find a safe position using log data
            max_attempts = 100
            safe_position_found = False
            
            for attempt in range(max_attempts):
                # Sample a random state from log data
                idx = np.random.randint(0, len(self.log_states))
                selected_log_idx = idx
                log_state = self.log_states[idx]
                
                # Use log data: altitude_agl -> z, roll, pitch, heading, altitude_rate
                z = log_state[0]  # altitude_agl from log
                # Clamp z to valid range
                z = np.clip(z, self.initial_position_range[0], min(self.initial_position_range[1], self.world_size[2] - 10))
                
                # Use longitude for x position (normalize to world coordinates)
                # Longitude is around 32.98, so we'll use it as offset
                lon_offset = self.log_longitude[selected_log_idx] - 32.98  # Center around 0
                x = lon_offset * 1000.0  # Scale longitude to meters (approximate)
                x = np.clip(x, -self.world_size[0]/2 + 50, self.world_size[0]/2 - 50)
                
                # Y position: random (since we don't have latitude in log)
                y = np.random.uniform(-self.world_size[1]/2 + 50, self.world_size[1]/2 - 50)
                
                initial_pos = np.array([x, y, z], dtype=np.float32)
                
                # Check if initial position is safe (no collision AND minimum distance from obstacles)
                collision, _ = self._check_collisions(initial_pos)
                min_obstacle_dist = self._get_min_obstacle_distance(initial_pos)
                
                # Safe if no collision and at least 30m away from obstacles
                if not collision and min_obstacle_dist >= 30.0:
                    self.position = initial_pos
                    safe_position_found = True
                    break
                else:
                    # Try to find nearby safe position
                    for offset_attempt in range(10):
                        offset_x = np.random.uniform(-100, 100)
                        offset_y = np.random.uniform(-100, 100)
                        offset_z = np.random.uniform(-20, 20)
                        test_pos = initial_pos + np.array([offset_x, offset_y, offset_z])
                        test_pos[2] = np.clip(test_pos[2], self.initial_position_range[0], 
                                            min(self.initial_position_range[1], self.world_size[2] - 10))
                        
                        collision_test, _ = self._check_collisions(test_pos)
                        min_dist_test = self._get_min_obstacle_distance(test_pos)
                        
                        if not collision_test and min_dist_test >= 30.0:
                            self.position = test_pos
                            safe_position_found = True
                            break
                    
                    if safe_position_found:
                        break
            
            # If still no safe position found, use random initialization
            if not safe_position_found:
                for attempt in range(max_attempts):
                    x = np.random.uniform(-self.world_size[0]/2 + 50, self.world_size[0]/2 - 50)
                    y = np.random.uniform(-self.world_size[1]/2 + 50, self.world_size[1]/2 - 50)
                    z = np.random.uniform(*self.initial_position_range)
                    initial_pos = np.array([x, y, z], dtype=np.float32)
                    
                    collision, _ = self._check_collisions(initial_pos)
                    min_obstacle_dist = self._get_min_obstacle_distance(initial_pos)
                    
                    if not collision and min_obstacle_dist >= 30.0:
                        self.position = initial_pos
                        safe_position_found = True
                        break
                
                if not safe_position_found:
                    # Fallback: use center, but ensure it's safe
                    self.position = np.array([0.0, 0.0, 100.0], dtype=np.float32)
                    # If center is not safe, move away from obstacles
                    min_dist = self._get_min_obstacle_distance(self.position)
                    if min_dist < 30.0:
                        # Move to a safe position
                        self.position = np.array([150.0, 150.0, 100.0], dtype=np.float32)
            
            # Get log state for attitude and velocity
            if selected_log_idx is not None:
                log_state = self.log_states[selected_log_idx]
            else:
                # Use random log state
                selected_log_idx = np.random.randint(0, len(self.log_states))
                log_state = self.log_states[selected_log_idx]
            
            # Initial velocity from log data
            # We have altitude_rate (vz), estimate vx, vy from heading
            heading_rad = np.deg2rad(log_state[3])  # heading
            # Estimate horizontal speed from altitude_rate (simplified)
            estimated_speed = abs(log_state[4]) * 2.0  # Rough estimate
            vx = estimated_speed * np.cos(heading_rad) if estimated_speed > 0 else 0.0
            vy = estimated_speed * np.sin(heading_rad) if estimated_speed > 0 else 0.0
            vz = log_state[4]  # altitude_rate
            
            self.velocity = np.array([vx, vy, vz], dtype=np.float32)
            # Clamp velocity
            speed = np.linalg.norm(self.velocity)
            if speed > self.max_speed:
                self.velocity = self.velocity / speed * self.max_speed
            
            # Initial attitude from log data
            self.attitude = np.array([
                log_state[1],  # roll
                log_state[2],  # pitch
                log_state[3]   # heading (yaw)
            ], dtype=np.float32)
            
            self.altitude_rate = log_state[4]  # altitude_rate
            
        else:
            # Random initialization (original code)
            max_attempts = 100
            for _ in range(max_attempts):
                x = np.random.uniform(-self.world_size[0]/2 + 50, self.world_size[0]/2 - 50)
                y = np.random.uniform(-self.world_size[1]/2 + 50, self.world_size[1]/2 - 50)
                z = np.random.uniform(*self.initial_position_range)
                initial_pos = np.array([x, y, z], dtype=np.float32)
                
                # Check if initial position is safe
                collision, _ = self._check_collisions(initial_pos)
                if not collision:
                    self.position = initial_pos
                    break
            else:
                # Fallback: use center
                self.position = np.array([0.0, 0.0, 100.0], dtype=np.float32)
            
            # Initial velocity
            self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            # Initial attitude
            self.attitude = np.array([
                np.random.uniform(-5.0, 5.0),  # roll
                np.random.uniform(-5.0, 5.0),  # pitch
                np.random.uniform(0.0, 360.0)  # yaw
            ], dtype=np.float32)
            
            self.altitude_rate = 0.0
        
        # Target position - behind the main obstacle (relative to helicopter)
        if self.target_position is None:
            if self.target_behind_obstacle and self.main_obstacle is not None:
                # Place target behind the main obstacle (relative to helicopter start position)
                obstacle_pos = self.main_obstacle.position
                heli_pos = self.position
                
                # Vector from helicopter to obstacle
                to_obstacle = obstacle_pos[:2] - heli_pos[:2]
                to_obstacle_norm = np.linalg.norm(to_obstacle)
                if to_obstacle_norm > 0:
                    to_obstacle = to_obstacle / to_obstacle_norm
                else:
                    to_obstacle = np.array([1.0, 0.0])
                
                # Place target behind obstacle (further from helicopter)
                # Distance: obstacle radius + safe margin (50m)
                distance_behind = self.main_obstacle.radius + 50.0
                target_xy = obstacle_pos[:2] + to_obstacle * distance_behind
                
                # Keep target in world bounds
                target_xy[0] = np.clip(target_xy[0], -self.world_size[0]/2 + 20, self.world_size[0]/2 - 20)
                target_xy[1] = np.clip(target_xy[1], -self.world_size[1]/2 + 20, self.world_size[1]/2 - 20)
                
                # Target height: similar to obstacle middle or helicopter altitude
                # Place target at a reasonable altitude (not too high, not too low)
                target_z = np.random.uniform(
                    max(obstacle_pos[2] + 20, self.initial_position_range[0]),
                    min(obstacle_pos[2] + self.main_obstacle.height - 20, self.initial_position_range[1])
                )
                target_z = np.clip(target_z, self.initial_position_range[0], self.world_size[2] - 10)
                
                self.target_position = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
                
                # Verify target is safe (not colliding with obstacle)
                collision, _ = self._check_collisions(self.target_position)
                if collision:
                    # If collision, place target slightly higher or adjust position
                    self.target_position[2] += 30.0
                    # Also move a bit further
                    self.target_position[:2] += to_obstacle * 20.0
            else:
                # Random target, far from obstacles
                max_attempts = 100
                for _ in range(max_attempts):
                    tx = np.random.uniform(-self.world_size[0]/2 + 50, self.world_size[0]/2 - 50)
                    ty = np.random.uniform(-self.world_size[1]/2 + 50, self.world_size[1]/2 - 50)
                    tz = np.random.uniform(*self.initial_position_range)
                    target = np.array([tx, ty, tz], dtype=np.float32)
                    
                    collision, _ = self._check_collisions(target)
                    if not collision:
                        self.target_position = target
                        break
                else:
                    self.target_position = np.array([200.0, 200.0, 100.0], dtype=np.float32)
        else:
            self.target_position = np.array(self.target_position, dtype=np.float32)
        
        self.last_distance_to_target = np.linalg.norm(self.position - self.target_position)
        self.step_count = 0
        self.episode_rewards = []
        
        state = self._get_state()
        info = {
            'position': self.position.copy(),
            'target': self.target_position.copy(),
            'distance_to_target': self.last_distance_to_target
        }
        
        return state, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.attitude,
            [self.altitude_rate]
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Obstacles are fixed (not moving), so no update needed
        
        roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd = action
        
        # Update attitude (simplified dynamics)
        self.attitude[0] += roll_cmd * 10.0 * self.dt  # Roll
        self.attitude[1] += pitch_cmd * 10.0 * self.dt  # Pitch
        self.attitude[2] += yaw_cmd * 20.0 * self.dt  # Yaw
        
        # Clamp attitude
        self.attitude[0] = np.clip(self.attitude[0], -90.0, 90.0)
        self.attitude[1] = np.clip(self.attitude[1], -90.0, 90.0)
        self.attitude[2] = self.attitude[2] % 360.0
        
        # Convert attitude to radians for calculations
        roll_rad = np.deg2rad(self.attitude[0])
        pitch_rad = np.deg2rad(self.attitude[1])
        yaw_rad = np.deg2rad(self.attitude[2])
        
        # Calculate acceleration based on attitude and throttle
        # Simplified helicopter dynamics
        forward_accel = throttle_cmd * 20.0 * np.cos(pitch_rad)
        lateral_accel = throttle_cmd * 20.0 * np.sin(roll_rad) * np.cos(pitch_rad)
        vertical_accel = throttle_cmd * 15.0 - 9.81  # Gravity compensation
        
        # Update velocity
        self.velocity[0] += forward_accel * np.cos(yaw_rad) * self.dt
        self.velocity[1] += forward_accel * np.sin(yaw_rad) * self.dt
        self.velocity[0] += lateral_accel * np.cos(yaw_rad + np.pi/2) * self.dt
        self.velocity[1] += lateral_accel * np.sin(yaw_rad + np.pi/2) * self.dt
        self.velocity[2] += vertical_accel * self.dt
        
        # Clamp velocity
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        # Update position
        self.position += self.velocity * self.dt
        
        # Update altitude rate
        self.altitude_rate = self.velocity[2]
        
        # Check collisions
        collision, collided_obstacle = self._check_collisions(self.position)
        
        # Calculate reward
        reward = 0.0
        
        # Collision penalty
        if collision:
            reward += self.collision_penalty
        else:
            # Progress reward (getting closer to target)
            distance_to_target = np.linalg.norm(self.position - self.target_position)
            progress = self.last_distance_to_target - distance_to_target
            reward += self.progress_reward * progress
            self.last_distance_to_target = distance_to_target
            
            # Goal reward
            if distance_to_target < 10.0:  # Within 10m of target
                reward += self.goal_reward
            
            # Obstacle avoidance (penalty for getting too close)
            min_obstacle_dist = self._get_min_obstacle_distance(self.position)
            if min_obstacle_dist < 20.0:
                reward += self.obstacle_penalty * (1.0 - min_obstacle_dist / 20.0)
            
            # Stability reward (small attitude angles)
            attitude_penalty = -0.01 * (abs(self.attitude[0]) + abs(self.attitude[1]))
            reward += attitude_penalty
        
        self.episode_rewards.append(reward)
        self.step_count += 1
        
        # Termination conditions
        terminated = collision
        truncated = (self.step_count >= self.max_episode_steps)
        
        info = {
            'position': self.position.copy(),
            'target': self.target_position.copy(),
            'distance_to_target': self.last_distance_to_target,
            'collision': collision,
            'collided_obstacle': collided_obstacle is not None if collision else False,
            'min_obstacle_distance': self._get_min_obstacle_distance(self.position),
            'episode_reward': sum(self.episode_rewards)
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            if self.fig is None:
                self.fig = plt.figure(figsize=(12, 10))
                self.ax = self.fig.add_subplot(111, projection='3d')
                plt.ion()
            
            self.ax.clear()
            
            # Draw obstacles
            for obstacle in self.obstacles:
                if obstacle.type == 'cylinder':
                    # Draw cylinder
                    z_bottom = obstacle.position[2]
                    z_top = obstacle.position[2] + obstacle.height
                    
                    # Draw cylinder sides
                    theta = np.linspace(0, 2*np.pi, 20)
                    x_circle = obstacle.position[0] + obstacle.radius * np.cos(theta)
                    y_circle = obstacle.position[1] + obstacle.radius * np.sin(theta)
                    
                    # Bottom circle
                    self.ax.plot(x_circle, y_circle, [z_bottom]*len(theta), 'r-', linewidth=2)
                    # Top circle
                    self.ax.plot(x_circle, y_circle, [z_top]*len(theta), 'r-', linewidth=2)
                    # Vertical lines
                    for i in range(0, len(theta), 4):
                        self.ax.plot(
                            [x_circle[i], x_circle[i]],
                            [y_circle[i], y_circle[i]],
                            [z_bottom, z_top],
                            'r-', linewidth=1, alpha=0.5
                        )
            
            # Draw helicopter (current position)
            self.ax.scatter(
                [self.position[0]], [self.position[1]], [self.position[2]],
                c='blue', marker='^', s=200, label='Helicopter'
            )
            
            # Draw target
            self.ax.scatter(
                [self.target_position[0]], [self.target_position[1]], [self.target_position[2]],
                c='green', marker='*', s=300, label='Target'
            )
            
            # Draw trajectory (last 50 positions)
            if hasattr(self, 'trajectory'):
                traj = np.array(self.trajectory[-50:])
                if len(traj) > 1:
                    self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.3, linewidth=1)
            
            # Store trajectory
            if not hasattr(self, 'trajectory'):
                self.trajectory = []
            self.trajectory.append(self.position.copy())
            if len(self.trajectory) > 1000:
                self.trajectory = self.trajectory[-1000:]
            
            # Set axis limits
            self.ax.set_xlim([-self.world_size[0]/2, self.world_size[0]/2])
            self.ax.set_ylim([-self.world_size[1]/2, self.world_size[1]/2])
            self.ax.set_zlim([0, self.world_size[2]])
            
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(f'3D Flight Environment - Step: {self.step_count}')
            self.ax.legend()
            
            plt.draw()
            plt.pause(0.01)
    
    def close(self):
        """Clean up environment resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    # Test environment
    print("Testing FlightControlEnv3D...")
    
    env = FlightControlEnv3D(
        world_size=(500.0, 500.0, 200.0),
        num_obstacles=1,  # Single fixed obstacle
        max_episode_steps=500,
        render_mode='human',
        moving_obstacles=False,  # Fixed obstacle
        obstacle_speed=0.0,
        target_behind_obstacle=True  # Target behind obstacle
    )
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial position: {info['position']}")
    print(f"Target position: {info['target']}")
    print(f"Distance to target: {info['distance_to_target']:.2f}m")
    print(f"Number of obstacles: {len(env.obstacles)}")
    
    # Test random steps
    print("\nRunning 50 random steps...")
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}: Reward={reward:.2f}, Distance={info['distance_to_target']:.2f}m, "
                  f"Pos={info['position'][:2]}")
        
        if terminated or truncated:
            print(f"Episode ended: {'Collision' if terminated else 'Timeout'}")
            break
        
        if env.render_mode == 'human':
            env.render()
    
    env.close()
    print("\nEnvironment test complete!")

