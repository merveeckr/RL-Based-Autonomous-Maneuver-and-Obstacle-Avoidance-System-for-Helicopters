"""
Improve Agent Behavior with Enhanced Log Data Usage
Shows how to better utilize log data for more realistic helicopter behavior.
"""

import numpy as np
import pandas as pd
from state_extractor import StateExtractor
from flight_env_3d import FlightControlEnv3D
from typing import Dict, List
import matplotlib.pyplot as plt


class LogDataEnhancer:
    """Enhance log data usage for better helicopter behavior."""
    
    def __init__(self, log_data_path: str = "fg_log2.csv"):
        """Initialize log data enhancer."""
        self.log_data_path = log_data_path
        self.extractor = StateExtractor(collision_threshold=2.0)
        self.df = self.extractor.load_data(log_data_path)
        self.state_df = self.extractor.extract_states(self.df)
    
    def analyze_log_data_quality(self) -> Dict:
        """
        Analyze quality and diversity of log data.
        
        Returns:
            Dictionary with analysis results
        """
        print("="*70)
        print("LOG DATA QUALITY ANALYSIS")
        print("="*70)
        
        analysis = {}
        
        # Data volume
        n_samples = len(self.state_df)
        duration = self.state_df['time'].max() - self.state_df['time'].min()
        
        print(f"\nData Volume:")
        print(f"  Total Samples: {n_samples}")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"  Sample Rate: {n_samples/duration:.2f} Hz")
        
        analysis['n_samples'] = n_samples
        analysis['duration'] = duration
        analysis['sample_rate'] = n_samples / duration
        
        # State diversity
        altitude_range = self.state_df['altitude_agl'].max() - self.state_df['altitude_agl'].min()
        roll_range = self.state_df['roll'].max() - self.state_df['roll'].min()
        pitch_range = self.state_df['pitch'].max() - self.state_df['pitch'].min()
        
        print(f"\nState Diversity:")
        print(f"  Altitude Range: {altitude_range:.2f}m")
        print(f"  Roll Range: {roll_range:.2f}°")
        print(f"  Pitch Range: {pitch_range:.2f}°")
        
        analysis['altitude_range'] = altitude_range
        analysis['roll_range'] = roll_range
        analysis['pitch_range'] = pitch_range
        
        # Flight phases
        # Identify different flight phases based on altitude_rate
        climbing = (self.state_df['altitude_rate'] > 5).sum()
        descending = (self.state_df['altitude_rate'] < -5).sum()
        level = ((self.state_df['altitude_rate'] >= -5) & (self.state_df['altitude_rate'] <= 5)).sum()
        
        print(f"\nFlight Phases:")
        print(f"  Climbing: {climbing} samples ({climbing/n_samples*100:.1f}%)")
        print(f"  Level Flight: {level} samples ({level/n_samples*100:.1f}%)")
        print(f"  Descending: {descending} samples ({descending/n_samples*100:.1f}%)")
        
        analysis['climbing_ratio'] = climbing / n_samples
        analysis['level_ratio'] = level / n_samples
        analysis['descending_ratio'] = descending / n_samples
        
        # Maneuver diversity
        high_roll = (self.state_df['roll'].abs() > 20).sum()
        high_pitch = (self.state_df['pitch'].abs() > 15).sum()
        
        print(f"\nManeuver Diversity:")
        print(f"  High Roll (>20°): {high_roll} samples ({high_roll/n_samples*100:.1f}%)")
        print(f"  High Pitch (>15°): {high_pitch} samples ({high_pitch/n_samples*100:.1f}%)")
        
        analysis['high_roll_ratio'] = high_roll / n_samples
        analysis['high_pitch_ratio'] = high_pitch / n_samples
        
        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*70)
        
        recommendations = []
        
        if n_samples < 1000:
            recommendations.append("⚠️  Low sample count - collect more log data")
        
        if altitude_range < 500:
            recommendations.append("⚠️  Limited altitude range - collect data at different altitudes")
        
        if analysis['high_roll_ratio'] < 0.1:
            recommendations.append("⚠️  Limited roll maneuvers - collect turning/ banking data")
        
        if analysis['high_pitch_ratio'] < 0.1:
            recommendations.append("⚠️  Limited pitch maneuvers - collect climbing/descending data")
        
        if analysis['level_ratio'] > 0.8:
            recommendations.append("⚠️  Too much level flight - collect more dynamic maneuvers")
        
        if len(recommendations) == 0:
            print("✅ Log data quality is good!")
        else:
            for rec in recommendations:
                print(rec)
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def suggest_enhanced_usage(self, analysis: Dict) -> Dict:
        """
        Suggest how to better use log data.
        
        Args:
            analysis: Analysis results from analyze_log_data_quality
            
        Returns:
            Dictionary with suggestions
        """
        print("\n" + "="*70)
        print("ENHANCED LOG DATA USAGE SUGGESTIONS")
        print("="*70)
        
        suggestions = {}
        
        # 1. Weighted sampling based on flight phase
        print("\n1. WEIGHTED STATE SAMPLING:")
        print("   - Sample more from interesting flight phases (turns, climbs)")
        print("   - Use log data to initialize in diverse scenarios")
        
        # 2. State-action pairs (if available)
        print("\n2. STATE-ACTION PAIRS:")
        print("   - If you have control inputs in log data, use them for:")
        print("     * Behavioral cloning (imitation learning)")
        print("     * Pre-training the agent")
        print("     * Reward shaping based on expert behavior")
        
        # 3. Curriculum learning
        print("\n3. CURRICULUM LEARNING:")
        print("   - Start with easy states (level flight)")
        print("   - Gradually introduce complex states (turns, maneuvers)")
        print("   - Use log data to define difficulty levels")
        
        # 4. Reward shaping
        print("\n4. REWARD SHAPING:")
        print("   - Add reward for matching log data statistics")
        print("   - Penalize unrealistic behaviors")
        print("   - Reward smooth, helicopter-like movements")
        
        # 5. Data augmentation
        print("\n5. DATA AUGMENTATION:")
        print("   - Add noise to log states for diversity")
        print("   - Interpolate between log states")
        print("   - Create synthetic states from log patterns")
        
        suggestions['weighted_sampling'] = True
        suggestions['curriculum_learning'] = True
        suggestions['reward_shaping'] = True
        
        return suggestions
    
    def create_enhanced_environment_config(self) -> Dict:
        """
        Create enhanced environment configuration using log data insights.
        
        Returns:
            Dictionary with enhanced config
        """
        print("\n" + "="*70)
        print("ENHANCED ENVIRONMENT CONFIGURATION")
        print("="*70)
        
        # Analyze log data to set realistic bounds
        altitude_mean = self.state_df['altitude_agl'].mean()
        altitude_std = self.state_df['altitude_agl'].std()
        
        roll_mean = abs(self.state_df['roll'].mean())
        roll_std = self.state_df['roll'].std()
        
        pitch_mean = abs(self.state_df['pitch'].mean())
        pitch_std = self.state_df['pitch'].std()
        
        config = {
            'initial_position_range': (
                max(altitude_mean - 2*altitude_std, 20),
                min(altitude_mean + 2*altitude_std, 200)
            ),
            'realistic_roll_range': (-roll_mean - 3*roll_std, roll_mean + 3*roll_std),
            'realistic_pitch_range': (-pitch_mean - 3*pitch_std, pitch_mean + 3*pitch_std),
            'target_altitude': altitude_mean,
            'use_log_data': True,
            'log_data_path': self.log_data_path
        }
        
        print(f"\nRecommended Configuration:")
        print(f"  Initial Altitude Range: {config['initial_position_range']}")
        print(f"  Realistic Roll Range: {config['realistic_roll_range']}")
        print(f"  Realistic Pitch Range: {config['realistic_pitch_range']}")
        print(f"  Target Altitude: {config['target_altitude']:.2f}m")
        
        return config


if __name__ == "__main__":
    print("="*70)
    print("LOG DATA ENHANCEMENT ANALYSIS")
    print("="*70)
    
    enhancer = LogDataEnhancer("fg_log2.csv")
    
    # Analyze quality
    analysis = enhancer.analyze_log_data_quality()
    
    # Get suggestions
    suggestions = enhancer.suggest_enhanced_usage(analysis)
    
    # Get enhanced config
    config = enhancer.create_enhanced_environment_config()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nTo improve agent behavior:")
    print("1. Analyze current behavior: python analyze_helicopter_behavior.py --model_path <model>")
    print("2. Collect more diverse log data if needed")
    print("3. Use weighted sampling from log data")
    print("4. Implement curriculum learning")
    print("5. Fine-tune reward function based on log statistics")

