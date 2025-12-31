"""
Reward Function Design Module
Designs and validates reward functions for RL training based on FlightGear data.

Reward Components:
R = R_altitude + R_stability + R_collision

Where:
- R_altitude: Reward for maintaining safe altitude
- R_stability: Penalty for excessive roll/pitch
- R_collision: Large penalty for collision
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from state_extractor import StateExtractor


class RewardDesigner:
    """Designs and validates reward functions for flight control RL."""
    
    def __init__(
        self,
        collision_threshold: float = 2.0,
        target_altitude: float = 100.0,
        altitude_tolerance: float = 20.0,
        max_roll: float = 30.0,
        max_pitch: float = 30.0,
        k1: float = 1.0,  # Altitude reward weight
        k2: float = 0.1,  # Roll penalty weight
        k3: float = 0.1,  # Pitch penalty weight
        collision_penalty: float = -100.0
    ):
        """
        Initialize reward designer.
        
        Args:
            collision_threshold: Minimum AGL altitude for collision (meters)
            target_altitude: Target altitude above ground (meters)
            altitude_tolerance: Acceptable deviation from target (meters)
            max_roll: Maximum acceptable roll angle (degrees)
            max_pitch: Maximum acceptable pitch angle (degrees)
            k1: Altitude reward coefficient
            k2: Roll penalty coefficient
            k3: Pitch penalty coefficient
            collision_penalty: Penalty for collision
        """
        self.collision_threshold = collision_threshold
        self.target_altitude = target_altitude
        self.altitude_tolerance = altitude_tolerance
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.collision_penalty = collision_penalty
    
    def compute_reward(
        self,
        altitude_agl: float,
        roll: float,
        pitch: float,
        altitude_rate: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for given state.
        
        Args:
            altitude_agl: Altitude above ground level (meters)
            roll: Roll angle (degrees)
            pitch: Pitch angle (degrees)
            altitude_rate: Rate of altitude change (m/s)
            
        Returns:
            Tuple of (total_reward, component_dict)
        """
        components = {}
        
        # Collision check
        if altitude_agl < self.collision_threshold:
            return self.collision_penalty, {'collision': self.collision_penalty}
        
        # Altitude reward (higher is better, with target preference)
        altitude_error = abs(altitude_agl - self.target_altitude)
        if altitude_error < self.altitude_tolerance:
            # Within tolerance: positive reward
            components['altitude'] = self.k1 * (1.0 - altitude_error / self.altitude_tolerance)
        else:
            # Outside tolerance: small penalty
            components['altitude'] = -self.k1 * 0.1 * (altitude_error / self.altitude_tolerance - 1.0)
        
        # Stability penalties (roll and pitch)
        roll_penalty = -self.k2 * (abs(roll) / self.max_roll) ** 2
        pitch_penalty = -self.k3 * (abs(pitch) / self.max_pitch) ** 2
        
        components['roll'] = roll_penalty
        components['pitch'] = pitch_penalty
        
        # Total reward
        total_reward = sum(components.values())
        
        return total_reward, components
    
    def compute_reward_batch(self, state_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rewards for entire state DataFrame.
        
        Args:
            state_df: DataFrame with state columns
            
        Returns:
            DataFrame with reward columns added
        """
        reward_df = state_df.copy()
        
        rewards = []
        reward_components = {
            'altitude_reward': [],
            'roll_penalty': [],
            'pitch_penalty': [],
            'collision_penalty': []
        }
        
        for idx, row in state_df.iterrows():
            reward, components = self.compute_reward(
                row['altitude_agl'],
                row['roll'],
                row['pitch'],
                row['altitude_rate']
            )
            
            rewards.append(reward)
            reward_components['altitude_reward'].append(components.get('altitude', 0.0))
            reward_components['roll_penalty'].append(components.get('roll', 0.0))
            reward_components['pitch_penalty'].append(components.get('pitch', 0.0))
            reward_components['collision_penalty'].append(components.get('collision', 0.0))
        
        reward_df['total_reward'] = rewards
        for key, values in reward_components.items():
            reward_df[key] = values
        
        return reward_df
    
    def visualize_reward_breakdown(self, reward_df: pd.DataFrame, save_path: str = "reward_breakdown.png"):
        """
        Visualize reward breakdown over time.
        
        Args:
            reward_df: DataFrame with reward columns
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Total reward over time
        axes[0].plot(reward_df['time'], reward_df['total_reward'], 'b-', linewidth=0.5, alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Total Reward Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Reward components
        axes[1].plot(reward_df['time'], reward_df['altitude_reward'], label='Altitude', alpha=0.7)
        axes[1].plot(reward_df['time'], reward_df['roll_penalty'], label='Roll Penalty', alpha=0.7)
        axes[1].plot(reward_df['time'], reward_df['pitch_penalty'], label='Pitch Penalty', alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Reward Component')
        axes[1].set_title('Reward Components Breakdown')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Altitude AGL with collision threshold
        axes[2].plot(reward_df['time'], reward_df['altitude_agl'], 'g-', linewidth=0.5, alpha=0.7)
        axes[2].axhline(y=self.collision_threshold, color='r', linestyle='--', label=f'Collision Threshold ({self.collision_threshold}m)')
        axes[2].axhline(y=self.target_altitude, color='b', linestyle='--', label=f'Target Altitude ({self.target_altitude}m)')
        axes[2].fill_between(reward_df['time'], 0, self.collision_threshold, alpha=0.2, color='red', label='Collision Zone')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Altitude AGL (m)')
        axes[2].set_title('Altitude Above Ground Level')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Reward breakdown saved to {save_path}")
        plt.close()
    
    def analyze_reward_statistics(self, reward_df: pd.DataFrame) -> Dict:
        """
        Analyze reward statistics.
        
        Args:
            reward_df: DataFrame with reward columns
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_reward': {
                'mean': reward_df['total_reward'].mean(),
                'std': reward_df['total_reward'].std(),
                'min': reward_df['total_reward'].min(),
                'max': reward_df['total_reward'].max(),
                'sum': reward_df['total_reward'].sum()
            },
            'collision_count': (reward_df['collision_penalty'] < 0).sum(),
            'positive_reward_ratio': (reward_df['total_reward'] > 0).sum() / len(reward_df),
            'altitude_stats': {
                'mean': reward_df['altitude_reward'].mean(),
                'std': reward_df['altitude_reward'].std()
            },
            'stability_stats': {
                'roll_penalty_mean': reward_df['roll_penalty'].mean(),
                'pitch_penalty_mean': reward_df['pitch_penalty'].mean()
            }
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("REWARD FUNCTION DESIGN AND VALIDATION")
    print("=" * 60)
    
    # Load and extract states
    extractor = StateExtractor(collision_threshold=2.0)
    df = extractor.load_data("fg_log2.csv")
    state_df = extractor.extract_states(df)
    
    # Design reward function
    reward_designer = RewardDesigner(
        collision_threshold=2.0,
        target_altitude=100.0,
        altitude_tolerance=20.0,
        k1=1.0,
        k2=0.1,
        k3=0.1,
        collision_penalty=-100.0
    )
    
    # Compute rewards
    print("\nComputing rewards...")
    reward_df = reward_designer.compute_reward_batch(state_df)
    
    # Analyze statistics
    print("\nReward Statistics:")
    stats = reward_designer.analyze_reward_statistics(reward_df)
    print(f"  Total Reward Mean: {stats['total_reward']['mean']:.4f}")
    print(f"  Total Reward Std: {stats['total_reward']['std']:.4f}")
    print(f"  Total Reward Sum: {stats['total_reward']['sum']:.4f}")
    print(f"  Collision Events: {stats['collision_count']}")
    print(f"  Positive Reward Ratio: {stats['positive_reward_ratio']*100:.2f}%")
    
    # Visualize
    print("\nGenerating visualizations...")
    reward_designer.visualize_reward_breakdown(reward_df, "reward_breakdown.png")
    
    print("\nReward design validation complete!")

