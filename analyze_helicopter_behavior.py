"""
Helicopter Behavior Analysis
Compares trained agent behavior with real FlightGear log data to validate
that the agent behaves like a real helicopter.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from flight_env_3d import FlightControlEnv3D
from state_extractor import StateExtractor
from typing import Dict, List, Tuple
import argparse


class HelicopterBehaviorAnalyzer:
    """Analyze and compare agent behavior with real helicopter data."""
    
    def __init__(self, model_path: str, log_data_path: str = "fg_log2.csv", 
                 filter_flight_phases: bool = True):
        """
        Initialize analyzer.
        
        Args:
            model_path: Path to trained PPO model
            log_data_path: Path to FlightGear log data
            filter_flight_phases: If True, filter out takeoff and landing phases
        """
        self.model_path = model_path
        self.log_data_path = log_data_path
        self.filter_flight_phases = filter_flight_phases
        
        # Load model
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            print(f"[OK] Model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load log data
        self.extractor = StateExtractor(collision_threshold=2.0)
        self.log_df = self.extractor.load_data(log_data_path)
        self.log_states = self.extractor.extract_states(self.log_df)
        print(f"[OK] Log data loaded: {len(self.log_states)} states")
        
        # Identify and filter flight phases
        if filter_flight_phases:
            print("\n[INFO] Identifying flight phases...")
            self.log_states = self.extractor.identify_flight_phases(self.log_states)
            
            # Show phase distribution
            phase_counts = self.log_states['flight_phase'].value_counts()
            print(f"[INFO] Flight phase distribution:")
            for phase, count in phase_counts.items():
                print(f"  {phase}: {count} samples ({count/len(self.log_states)*100:.1f}%)")
            
            # Filter to cruise only
            print("\n[INFO] Filtering to cruise (level flight) phase only...")
            original_count = len(self.log_states)
            self.log_states = self.extractor.filter_cruise_phase(self.log_states)
            filtered_count = len(self.log_states)
            print(f"[OK] Filtered: {original_count} -> {filtered_count} samples "
                  f"({filtered_count/original_count*100:.1f}% remaining)")
        
        # Statistics from filtered log data
        self.log_stats = self._compute_log_statistics()
        
    def _compute_log_statistics(self) -> Dict:
        """Compute statistics from log data."""
        stats = {
            'altitude_agl': {
                'mean': self.log_states['altitude_agl'].mean(),
                'std': self.log_states['altitude_agl'].std(),
                'min': self.log_states['altitude_agl'].min(),
                'max': self.log_states['altitude_agl'].max()
            },
            'roll': {
                'mean': self.log_states['roll'].mean(),
                'std': self.log_states['roll'].std(),
                'min': self.log_states['roll'].min(),
                'max': self.log_states['roll'].max()
            },
            'pitch': {
                'mean': self.log_states['pitch'].mean(),
                'std': self.log_states['pitch'].std(),
                'min': self.log_states['pitch'].min(),
                'max': self.log_states['pitch'].max()
            },
            'altitude_rate': {
                'mean': self.log_states['altitude_rate'].mean(),
                'std': self.log_states['altitude_rate'].std(),
                'min': self.log_states['altitude_rate'].min(),
                'max': self.log_states['altitude_rate'].max()
            }
        }
        return stats
    
    def collect_agent_trajectory(self, n_episodes: int = 10, max_steps: int = 1000) -> pd.DataFrame:
        """
        Collect agent trajectory data.
        
        Args:
            n_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            
        Returns:
            DataFrame with agent behavior data
        """
        env = FlightControlEnv3D(
            world_size=(500.0, 500.0, 200.0),
            num_obstacles=1,
            max_episode_steps=max_steps,
            use_log_data=True,
            log_data_path=self.log_data_path,
            moving_obstacles=False,
            target_behind_obstacle=True,
            render_mode=None
        )
        
        all_data = []
        
        print(f"\nCollecting agent trajectories ({n_episodes} episodes)...")
        for episode in range(n_episodes):
            obs, info = env.reset(seed=episode * 42)
            episode_data = []
            
            for step in range(max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Extract state information
                position = info['position']
                state = obs  # [x, y, z, vx, vy, vz, roll, pitch, yaw, altitude_rate]
                
                episode_data.append({
                    'episode': episode,
                    'step': step,
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],  # altitude_agl equivalent
                    'vx': state[3],
                    'vy': state[4],
                    'vz': state[5],
                    'roll': state[6],
                    'pitch': state[7],
                    'yaw': state[8],
                    'altitude_rate': state[9],
                    'reward': reward,
                    'distance_to_target': info['distance_to_target']
                })
                
                if terminated or truncated:
                    break
            
            all_data.extend(episode_data)
            if (episode + 1) % 5 == 0:
                print(f"  Completed {episode + 1}/{n_episodes} episodes")
        
        env.close()
        
        agent_df = pd.DataFrame(all_data)
        print(f"[OK] Collected {len(agent_df)} data points from agent")
        
        return agent_df
    
    def compare_statistics(self, agent_df: pd.DataFrame) -> Dict:
        """
        Compare agent statistics with log data statistics.
        Focus on technical characteristics: angles, rates, velocities (NOT altitude).
        
        Args:
            agent_df: DataFrame with agent behavior data
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*70)
        print("STATISTICAL COMPARISON: Agent vs Real Helicopter")
        print("(Focus: Technical Characteristics - Angles, Rates, Velocities)")
        print("="*70)
        
        comparisons = {}
        
        # Compare roll - FIXED similarity calculation
        agent_roll_mean = agent_df['roll'].abs().mean()
        agent_roll_std = agent_df['roll'].std()
        log_roll_mean = abs(self.log_stats['roll']['mean'])
        log_roll_std = self.log_stats['roll']['std']
        
        # Fixed similarity: use std as reference scale, clamp to [0, 1]
        roll_mean_diff = abs(agent_roll_mean - log_roll_mean)
        roll_std_ref = max(log_roll_std, 5.0)  # Minimum reference
        roll_similarity = max(0.0, min(1.0, 1.0 - roll_mean_diff / roll_std_ref))
        
        roll_std_similarity = max(0.0, min(1.0, 1.0 - abs(agent_roll_std - log_roll_std) / max(log_roll_std, 5.0)))
        
        print(f"\nRoll Angle:")
        print(f"  Real Helicopter: mean={log_roll_mean:.2f}°, std={log_roll_std:.2f}°")
        print(f"  Agent:          mean={agent_roll_mean:.2f}°, std={agent_roll_std:.2f}°")
        print(f"  Similarity:     {roll_similarity*100:.1f}% (mean), {roll_std_similarity*100:.1f}% (std)")
        
        comparisons['roll'] = {
            'agent_mean': agent_roll_mean,
            'agent_std': agent_roll_std,
            'log_mean': log_roll_mean,
            'log_std': log_roll_std,
            'similarity': roll_similarity,
            'std_similarity': roll_std_similarity
        }
        
        # Compare pitch - FIXED similarity calculation
        agent_pitch_mean = agent_df['pitch'].abs().mean()
        agent_pitch_std = agent_df['pitch'].std()
        log_pitch_mean = abs(self.log_stats['pitch']['mean'])
        log_pitch_std = self.log_stats['pitch']['std']
        
        pitch_mean_diff = abs(agent_pitch_mean - log_pitch_mean)
        pitch_std_ref = max(log_pitch_std, 3.0)  # Minimum reference
        pitch_similarity = max(0.0, min(1.0, 1.0 - pitch_mean_diff / pitch_std_ref))
        
        pitch_std_similarity = max(0.0, min(1.0, 1.0 - abs(agent_pitch_std - log_pitch_std) / max(log_pitch_std, 3.0)))
        
        print(f"\nPitch Angle:")
        print(f"  Real Helicopter: mean={log_pitch_mean:.2f}°, std={log_pitch_std:.2f}°")
        print(f"  Agent:          mean={agent_pitch_mean:.2f}°, std={agent_pitch_std:.2f}°")
        print(f"  Similarity:     {pitch_similarity*100:.1f}% (mean), {pitch_std_similarity*100:.1f}% (std)")
        
        comparisons['pitch'] = {
            'agent_mean': agent_pitch_mean,
            'agent_std': agent_pitch_std,
            'log_mean': log_pitch_mean,
            'log_std': log_pitch_std,
            'similarity': pitch_similarity,
            'std_similarity': pitch_std_similarity
        }
        
        # Compare altitude_rate - FIXED similarity calculation
        agent_rate_mean = agent_df['altitude_rate'].mean()
        agent_rate_std = agent_df['altitude_rate'].std()
        log_rate_mean = self.log_stats['altitude_rate']['mean']
        log_rate_std = self.log_stats['altitude_rate']['std']
        
        rate_mean_diff = abs(agent_rate_mean - log_rate_mean)
        rate_std_ref = max(log_rate_std, 5.0)  # Minimum reference
        rate_similarity = max(0.0, min(1.0, 1.0 - rate_mean_diff / rate_std_ref))
        
        rate_std_similarity = max(0.0, min(1.0, 1.0 - abs(agent_rate_std - log_rate_std) / max(log_rate_std, 5.0)))
        
        print(f"\nAltitude Rate:")
        print(f"  Real Helicopter: mean={log_rate_mean:.2f}m/s, std={log_rate_std:.2f}m/s")
        print(f"  Agent:          mean={agent_rate_mean:.2f}m/s, std={agent_rate_std:.2f}m/s")
        print(f"  Similarity:     {rate_similarity*100:.1f}% (mean), {rate_std_similarity*100:.1f}% (std)")
        
        comparisons['altitude_rate'] = {
            'agent_mean': agent_rate_mean,
            'agent_std': agent_rate_std,
            'log_mean': log_rate_mean,
            'log_std': log_rate_std,
            'similarity': rate_similarity,
            'std_similarity': rate_std_similarity
        }
        
        # Compare velocity (speed) - NEW
        agent_df['speed'] = np.sqrt(agent_df['vx']**2 + agent_df['vy']**2 + agent_df['vz']**2)
        agent_speed_mean = agent_df['speed'].mean()
        agent_speed_std = agent_df['speed'].std()
        
        # Estimate speed from log data (from altitude_rate and heading changes)
        # This is approximate since we don't have direct velocity in log
        log_estimated_speed = abs(self.log_stats['altitude_rate']['mean']) * 2.0  # Rough estimate
        log_estimated_speed_std = self.log_stats['altitude_rate']['std'] * 2.0
        
        speed_diff = abs(agent_speed_mean - log_estimated_speed)
        speed_ref = max(log_estimated_speed_std, 10.0)
        speed_similarity = max(0.0, min(1.0, 1.0 - speed_diff / speed_ref))
        
        print(f"\nSpeed (Estimated):")
        print(f"  Real Helicopter (est): mean={log_estimated_speed:.2f}m/s, std={log_estimated_speed_std:.2f}m/s")
        print(f"  Agent:                  mean={agent_speed_mean:.2f}m/s, std={agent_speed_std:.2f}m/s")
        print(f"  Similarity:             {speed_similarity*100:.1f}%")
        
        comparisons['speed'] = {
            'agent_mean': agent_speed_mean,
            'agent_std': agent_speed_std,
            'log_mean': log_estimated_speed,
            'log_std': log_estimated_speed_std,
            'similarity': speed_similarity
        }
        
        # Compare angular rates (from attitude changes) - NEW
        # Calculate angular rates from agent data
        agent_df_sorted = agent_df.sort_values(['episode', 'step'])
        agent_df_sorted['roll_rate'] = agent_df_sorted.groupby('episode')['roll'].diff().abs()
        agent_df_sorted['pitch_rate'] = agent_df_sorted.groupby('episode')['pitch'].diff().abs()
        
        agent_roll_rate_mean = agent_df_sorted['roll_rate'].mean() / 0.1  # per second (dt=0.1)
        agent_pitch_rate_mean = agent_df_sorted['pitch_rate'].mean() / 0.1
        
        # Log data doesn't have direct angular rates, but we can estimate from std
        # Higher std = more variation = higher rates
        log_roll_rate_est = self.log_stats['roll']['std'] * 0.5  # Rough estimate
        log_pitch_rate_est = self.log_stats['pitch']['std'] * 0.5
        
        roll_rate_diff = abs(agent_roll_rate_mean - log_roll_rate_est)
        roll_rate_ref = max(log_roll_rate_est, 2.0)
        roll_rate_similarity = max(0.0, min(1.0, 1.0 - roll_rate_diff / roll_rate_ref))
        
        pitch_rate_diff = abs(agent_pitch_rate_mean - log_pitch_rate_est)
        pitch_rate_ref = max(log_pitch_rate_est, 2.0)
        pitch_rate_similarity = max(0.0, min(1.0, 1.0 - pitch_rate_diff / pitch_rate_ref))
        
        print(f"\nAngular Rates (Estimated):")
        print(f"  Roll Rate:")
        print(f"    Real Helicopter (est): {log_roll_rate_est:.2f}°/s")
        print(f"    Agent:                  {agent_roll_rate_mean:.2f}°/s")
        print(f"    Similarity:             {roll_rate_similarity*100:.1f}%")
        print(f"  Pitch Rate:")
        print(f"    Real Helicopter (est): {log_pitch_rate_est:.2f}°/s")
        print(f"    Agent:                  {agent_pitch_rate_mean:.2f}°/s")
        print(f"    Similarity:             {pitch_rate_similarity*100:.1f}%")
        
        comparisons['roll_rate'] = {
            'agent_mean': agent_roll_rate_mean,
            'log_mean': log_roll_rate_est,
            'similarity': roll_rate_similarity
        }
        comparisons['pitch_rate'] = {
            'agent_mean': agent_pitch_rate_mean,
            'log_mean': log_pitch_rate_est,
            'similarity': pitch_rate_similarity
        }
        
        # Overall similarity score - ONLY technical characteristics (NO altitude)
        overall_similarity = (
            roll_similarity * 0.2 +
            roll_std_similarity * 0.1 +
            pitch_similarity * 0.2 +
            pitch_std_similarity * 0.1 +
            rate_similarity * 0.15 +
            rate_std_similarity * 0.1 +
            speed_similarity * 0.1 +
            roll_rate_similarity * 0.075 +
            pitch_rate_similarity * 0.075
        )
        
        print(f"\n{'='*70}")
        print(f"OVERALL SIMILARITY SCORE (Technical): {overall_similarity*100:.1f}%")
        print(f"{'='*70}")
        
        if overall_similarity > 0.7:
            print("[SUCCESS] Agent technical behavior is similar to real helicopter!")
        elif overall_similarity > 0.5:
            print("[WARNING] Agent technical behavior is somewhat similar, but could be improved.")
        else:
            print("[ALERT] Agent technical behavior differs from real helicopter.")
        
        comparisons['overall_similarity'] = overall_similarity
        
        return comparisons
    
    def analyze_flight_dynamics(self, agent_df: pd.DataFrame) -> Dict:
        """
        Analyze flight dynamics patterns.
        
        Args:
            agent_df: DataFrame with agent behavior data
            
        Returns:
            Dictionary with dynamics analysis
        """
        print("\n" + "="*70)
        print("FLIGHT DYNAMICS ANALYSIS")
        print("="*70)
        
        # Speed analysis
        agent_df['speed'] = np.sqrt(agent_df['vx']**2 + agent_df['vy']**2 + agent_df['vz']**2)
        avg_speed = agent_df['speed'].mean()
        max_speed = agent_df['speed'].max()
        
        print(f"\nSpeed Analysis:")
        print(f"  Average Speed: {avg_speed:.2f} m/s")
        print(f"  Max Speed: {max_speed:.2f} m/s")
        
        # Attitude stability
        roll_stability = agent_df['roll'].abs().mean()
        pitch_stability = agent_df['pitch'].abs().mean()
        
        print(f"\nAttitude Stability:")
        print(f"  Average |Roll|: {roll_stability:.2f}°")
        print(f"  Average |Pitch|: {pitch_stability:.2f}°")
        print(f"  {'Good stability' if roll_stability < 10 and pitch_stability < 10 else 'Needs improvement'}")
        
        # Altitude control
        altitude_variance = agent_df['z'].std()
        print(f"\nAltitude Control:")
        print(f"  Altitude Std Dev: {altitude_variance:.2f}m")
        print(f"  {'Good control' if altitude_variance < 50 else 'Needs improvement'}")
        
        # Action smoothness (check action changes)
        # This would require storing actions, but we can infer from state changes
        
        return {
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'roll_stability': roll_stability,
            'pitch_stability': pitch_stability,
            'altitude_variance': altitude_variance
        }
    
    def visualize_comparison(self, agent_df: pd.DataFrame, save_dir: str = "./behavior_analysis/"):
        """
        Create visualization comparing agent and real helicopter behavior.
        Focus on technical characteristics.
        
        Args:
            agent_df: DataFrame with agent behavior data
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Roll comparison (NOT altitude)
        ax = axes[0, 0]
        ax.hist(self.log_states['roll'], bins=50, alpha=0.5, label='Real Helicopter', color='blue')
        ax.hist(agent_df['roll'], bins=50, alpha=0.5, label='Agent', color='red')
        ax.set_xlabel('Roll Angle (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title('Roll Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Pitch comparison
        ax = axes[0, 1]
        ax.hist(self.log_states['pitch'], bins=50, alpha=0.5, label='Real Helicopter', color='blue')
        ax.hist(agent_df['pitch'], bins=50, alpha=0.5, label='Agent', color='red')
        ax.set_xlabel('Pitch Angle (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title('Pitch Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Altitude rate comparison
        ax = axes[0, 2]
        ax.hist(self.log_states['altitude_rate'], bins=50, alpha=0.5, label='Real Helicopter', color='blue')
        ax.hist(agent_df['altitude_rate'], bins=50, alpha=0.5, label='Agent', color='red')
        ax.set_xlabel('Altitude Rate (m/s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Altitude Rate Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Speed comparison (NEW)
        ax = axes[1, 0]
        agent_df['speed'] = np.sqrt(agent_df['vx']**2 + agent_df['vy']**2 + agent_df['vz']**2)
        # Estimate speed from log (approximate)
        log_speed_est = abs(self.log_states['altitude_rate']) * 2.0
        ax.hist(log_speed_est, bins=50, alpha=0.5, label='Real Helicopter (est)', color='blue')
        ax.hist(agent_df['speed'], bins=50, alpha=0.5, label='Agent', color='red')
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Speed Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Attitude stability over time
        ax = axes[1, 1]
        sample_episode = agent_df[agent_df['episode'] == 0]
        ax.plot(sample_episode['step'], sample_episode['roll'].abs(), 'r-', label='Agent |Roll|', alpha=0.7)
        ax.plot(sample_episode['step'], sample_episode['pitch'].abs(), 'b-', label='Agent |Pitch|', alpha=0.7)
        # Log data sample
        log_sample = self.log_states.iloc[:len(sample_episode)]
        ax.plot(range(len(log_sample)), log_sample['roll'].abs(), 'r--', label='Real |Roll|', alpha=0.5)
        ax.plot(range(len(log_sample)), log_sample['pitch'].abs(), 'b--', label='Real |Pitch|', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Attitude Stability Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Angular rate comparison (NEW)
        ax = axes[1, 2]
        agent_df_sorted = agent_df.sort_values(['episode', 'step'])
        agent_df_sorted['roll_rate'] = agent_df_sorted.groupby('episode')['roll'].diff().abs() / 0.1
        agent_df_sorted['pitch_rate'] = agent_df_sorted.groupby('episode')['pitch'].diff().abs() / 0.1
        
        ax.hist(agent_df_sorted['roll_rate'].dropna(), bins=30, alpha=0.5, label='Agent Roll Rate', color='red')
        ax.hist(agent_df_sorted['pitch_rate'].dropna(), bins=30, alpha=0.5, label='Agent Pitch Rate', color='blue')
        ax.set_xlabel('Angular Rate (degrees/s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Angular Rate Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, "behavior_comparison.png")
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Visualization saved to: {save_path}")
        plt.close()
    
    def generate_report(self, agent_df: pd.DataFrame, comparisons: Dict, 
                       dynamics: Dict, save_path: str = "./behavior_analysis/behavior_report.txt"):
        """
        Generate detailed behavior analysis report.
        
        Args:
            agent_df: DataFrame with agent behavior data
            comparisons: Comparison statistics
            dynamics: Flight dynamics analysis
            save_path: Path to save report
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        report = f"""
HELICOPTER BEHAVIOR ANALYSIS REPORT
===================================
Generated from: {self.model_path}
Log Data: {self.log_data_path}

FOCUS: Technical Characteristics (Angles, Rates, Velocities)
NOT INCLUDED: Altitude (different operational scales)

1. STATISTICAL COMPARISON
-------------------------
Overall Similarity Score: {comparisons['overall_similarity']*100:.1f}%

Roll Angle:
  Real Helicopter: mean={comparisons['roll']['log_mean']:.2f}°, std={comparisons['roll']['log_std']:.2f}°
  Agent:          mean={comparisons['roll']['agent_mean']:.2f}°, std={comparisons['roll']['agent_std']:.2f}°
  Similarity:     {comparisons['roll']['similarity']*100:.1f}% (mean), {comparisons['roll']['std_similarity']*100:.1f}% (std)

Pitch Angle:
  Real Helicopter: mean={comparisons['pitch']['log_mean']:.2f}°, std={comparisons['pitch']['log_std']:.2f}°
  Agent:          mean={comparisons['pitch']['agent_mean']:.2f}°, std={comparisons['pitch']['agent_std']:.2f}°
  Similarity:     {comparisons['pitch']['similarity']*100:.1f}% (mean), {comparisons['pitch']['std_similarity']*100:.1f}% (std)

Altitude Rate:
  Real Helicopter: mean={comparisons['altitude_rate']['log_mean']:.2f}m/s, std={comparisons['altitude_rate']['log_std']:.2f}m/s
  Agent:          mean={comparisons['altitude_rate']['agent_mean']:.2f}m/s, std={comparisons['altitude_rate']['agent_std']:.2f}m/s
  Similarity:     {comparisons['altitude_rate']['similarity']*100:.1f}% (mean), {comparisons['altitude_rate']['std_similarity']*100:.1f}% (std)

Speed:
  Real Helicopter (est): mean={comparisons['speed']['log_mean']:.2f}m/s
  Agent:                 mean={comparisons['speed']['agent_mean']:.2f}m/s
  Similarity:            {comparisons['speed']['similarity']*100:.1f}%

Angular Rates:
  Roll Rate:
    Real Helicopter (est): {comparisons['roll_rate']['log_mean']:.2f}°/s
    Agent:                 {comparisons['roll_rate']['agent_mean']:.2f}°/s
    Similarity:            {comparisons['roll_rate']['similarity']*100:.1f}%
  Pitch Rate:
    Real Helicopter (est): {comparisons['pitch_rate']['log_mean']:.2f}°/s
    Agent:                 {comparisons['pitch_rate']['agent_mean']:.2f}°/s
    Similarity:            {comparisons['pitch_rate']['similarity']*100:.1f}%

2. FLIGHT DYNAMICS
------------------
Average Speed: {dynamics['avg_speed']:.2f} m/s
Max Speed: {dynamics['max_speed']:.2f} m/s
Roll Stability (avg |roll|): {dynamics['roll_stability']:.2f}°
Pitch Stability (avg |pitch|): {dynamics['pitch_stability']:.2f}°
Altitude Variance: {dynamics['altitude_variance']:.2f}m

3. BEHAVIOR ASSESSMENT
----------------------
"""
        
        overall = comparisons['overall_similarity']
        if overall > 0.7:
            report += "✅ Agent technical behavior is SIMILAR to real helicopter.\n"
            report += "   The agent demonstrates realistic flight characteristics.\n"
        elif overall > 0.5:
            report += "⚠️  Agent technical behavior is MODERATELY similar to real helicopter.\n"
            report += "   Some improvements needed for more realistic behavior.\n"
        else:
            report += "❌ Agent technical behavior DIFFERS from real helicopter.\n"
            report += "   Significant improvements needed.\n"
        
        report += f"""
4. RECOMMENDATIONS
------------------
"""
        
        if comparisons['roll']['similarity'] < 0.7:
            report += "- Improve roll angle control for more realistic banking\n"
        
        if comparisons['pitch']['similarity'] < 0.7:
            report += "- Improve pitch angle control for more realistic climbing/descending\n"
        
        if comparisons['altitude_rate']['similarity'] < 0.7:
            report += "- Improve altitude rate control for smoother altitude changes\n"
        
        if dynamics['roll_stability'] > 15 or dynamics['pitch_stability'] > 15:
            report += "- Improve attitude stability (reduce excessive roll/pitch)\n"
        
        report += """
5. LOG DATA USAGE
-----------------
Current log data provides:
- Realistic initial states (attitude, velocity)
- Real helicopter flight patterns for comparison
- Validation baseline for agent behavior

To improve agent behavior:
- Use more diverse log data (different flight scenarios)
- Increase training time with log data initialization
- Fine-tune reward function to encourage realistic dynamics
- Focus on matching angular rates and attitude stability
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Report saved to: {save_path}")
        print("\nReport Preview:")
        print(report[:500] + "...")


def main():
    parser = argparse.ArgumentParser(description='Analyze helicopter behavior')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PPO model')
    parser.add_argument('--log_data', type=str, default='fg_log2.csv',
                       help='Path to FlightGear log data')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of episodes to analyze')
    parser.add_argument('--output_dir', type=str, default='./behavior_analysis/',
                       help='Output directory for analysis results')
    parser.add_argument('--filter_phases', action='store_true', default=True,
                       help='Filter out takeoff and landing phases (default: True)')
    parser.add_argument('--no_filter_phases', dest='filter_phases', action='store_false',
                       help='Do not filter flight phases')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HelicopterBehaviorAnalyzer(args.model_path, args.log_data, 
                                         filter_flight_phases=args.filter_phases)
    
    # Collect agent trajectories
    agent_df = analyzer.collect_agent_trajectory(n_episodes=args.n_episodes)
    
    # Compare statistics
    comparisons = analyzer.compare_statistics(agent_df)
    
    # Analyze flight dynamics
    dynamics = analyzer.analyze_flight_dynamics(agent_df)
    
    # Visualize
    analyzer.visualize_comparison(agent_df, save_dir=args.output_dir)
    
    # Generate report
    report_path = os.path.join(args.output_dir, "behavior_report.txt")
    analyzer.generate_report(agent_df, comparisons, dynamics, report_path)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

