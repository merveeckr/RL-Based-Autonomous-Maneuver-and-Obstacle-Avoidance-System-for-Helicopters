"""
State Extraction Module
Extracts state vectors from FlightGear telemetry logs for RL training.

State Vector (5D):
- altitude_agl: Altitude above ground level (meters)
- roll: Roll angle (degrees)
- pitch: Pitch angle (degrees)
- heading: Heading/Yaw angle (degrees)
- altitude_rate: Rate of altitude change (m/s)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class StateExtractor:
    """Extracts and validates state vectors from FlightGear telemetry data."""
    
    def __init__(self, collision_threshold: float = 2.0):
        """
        Initialize state extractor.
        
        Args:
            collision_threshold: Minimum altitude above ground to avoid collision (meters)
        """
        self.collision_threshold = collision_threshold
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load FlightGear telemetry CSV file."""
        df = pd.read_csv(csv_path)
        return df
    
    def extract_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract state vectors from raw telemetry data.
        
        Args:
            df: DataFrame with columns: Time, Altitude, Roll, Pitch, Heading, GroundAltitude
            
        Returns:
            DataFrame with extracted state vectors
        """
        df = df.copy()
        
        # Calculate altitude above ground level (AGL)
        df['altitude_agl'] = df['Altitude'] - df['GroundAltitude']
        
        # Calculate altitude rate (derivative)
        # Assuming time steps are ~1 second (adjust if needed)
        time_diff = df['Time'].diff().fillna(1.0)  # Default to 1 second if missing
        altitude_diff = df['Altitude'].diff()
        df['altitude_rate'] = altitude_diff / time_diff
        
        # Normalize heading to [0, 360) range (already in degrees)
        df['heading'] = df['Heading'] % 360
        
        # Create state DataFrame
        state_df = pd.DataFrame({
            'time': df['Time'],
            'altitude_agl': df['altitude_agl'],
            'roll': df['Roll'],
            'pitch': df['Pitch'],
            'heading': df['heading'],
            'altitude_rate': df['altitude_rate']
        })
        
        # Fill NaN values (first row altitude_rate will be NaN)
        state_df['altitude_rate'] = state_df['altitude_rate'].fillna(0.0)
        
        return state_df
    
    def detect_collisions(self, state_df: pd.DataFrame) -> pd.Series:
        """
        Detect collision events based on altitude AGL.
        
        Args:
            state_df: DataFrame with altitude_agl column
            
        Returns:
            Boolean series indicating collision events
        """
        return state_df['altitude_agl'] < self.collision_threshold
    
    def get_state_vector(self, state_df: pd.DataFrame, index: int) -> np.ndarray:
        """
        Extract state vector as numpy array for RL agent.
        
        Args:
            state_df: DataFrame with state columns
            index: Row index
            
        Returns:
            State vector as numpy array [altitude_agl, roll, pitch, heading, altitude_rate]
        """
        row = state_df.iloc[index]
        return np.array([
            row['altitude_agl'],
            row['roll'],
            row['pitch'],
            row['heading'],
            row['altitude_rate']
        ])
    
    def get_state_bounds(self, state_df: pd.DataFrame) -> dict:
        """
        Get min/max bounds for each state dimension (for normalization).
        
        Args:
            state_df: DataFrame with state columns
            
        Returns:
            Dictionary with min/max bounds for each state dimension
        """
        bounds = {
            'altitude_agl': (state_df['altitude_agl'].min(), state_df['altitude_agl'].max()),
            'roll': (state_df['roll'].min(), state_df['roll'].max()),
            'pitch': (state_df['pitch'].min(), state_df['pitch'].max()),
            'heading': (state_df['heading'].min(), state_df['heading'].max()),
            'altitude_rate': (state_df['altitude_rate'].min(), state_df['altitude_rate'].max())
        }
        return bounds
    
    def validate_state(self, state: np.ndarray) -> bool:
        """
        Validate state vector (check for NaN, Inf, etc.).
        
        Args:
            state: State vector array
            
        Returns:
            True if state is valid
        """
        return np.all(np.isfinite(state))
    
    def identify_flight_phases(self, state_df: pd.DataFrame, 
                              altitude_rate_threshold: float = 5.0,
                              min_cruise_altitude: float = 50.0) -> pd.DataFrame:
        """
        Identify flight phases: takeoff, cruise, landing.
        
        Args:
            state_df: DataFrame with state columns
            altitude_rate_threshold: Maximum altitude_rate for level flight (m/s)
            min_cruise_altitude: Minimum altitude to consider as cruise (meters)
            
        Returns:
            DataFrame with 'flight_phase' column added
        """
        state_df = state_df.copy()
        
        # Identify phases based on altitude_rate and altitude
        conditions = [
            # Takeoff: high positive altitude_rate OR low altitude with positive rate
            (state_df['altitude_rate'] > altitude_rate_threshold) | 
            ((state_df['altitude_agl'] < min_cruise_altitude) & (state_df['altitude_rate'] > 0)),
            
            # Landing: high negative altitude_rate OR low altitude with negative rate
            (state_df['altitude_rate'] < -altitude_rate_threshold) |
            ((state_df['altitude_agl'] < min_cruise_altitude) & (state_df['altitude_rate'] < 0)),
            
            # Cruise: everything else (level flight)
            True
        ]
        
        choices = ['takeoff', 'landing', 'cruise']
        state_df['flight_phase'] = np.select(conditions, choices, default='cruise')
        
        return state_df
    
    def filter_cruise_phase(self, state_df: pd.DataFrame, 
                           altitude_rate_threshold: float = 5.0,
                           min_altitude: float = 50.0) -> pd.DataFrame:
        """
        Filter to keep only cruise (level flight) phase.
        Removes takeoff and landing phases.
        
        Args:
            state_df: DataFrame with state columns
            altitude_rate_threshold: Maximum altitude_rate for level flight (m/s)
            min_altitude: Minimum altitude to consider (meters)
        
        Returns:
            Filtered DataFrame with only cruise/level flight data
        """
        # Filter: altitude_rate should be low (level flight)
        # AND altitude should be above minimum (already in flight)
        filtered = state_df[
            (state_df['altitude_rate'].abs() <= altitude_rate_threshold) &
            (state_df['altitude_agl'] >= min_altitude)
        ].copy()
        
        return filtered



if __name__ == "__main__":
    # Example usage
    extractor = StateExtractor(collision_threshold=2.0)
    
    # Load data
    df = extractor.load_data("fg_log2.csv")
    print(f"Loaded {len(df)} data points")
    
    # Extract states
    state_df = extractor.extract_states(df)
    print(f"\nState DataFrame shape: {state_df.shape}")
    print(f"\nState columns: {state_df.columns.tolist()}")
    print(f"\nFirst 5 states:")
    print(state_df.head())
    
    # Detect collisions
    collisions = extractor.detect_collisions(state_df)
    collision_count = collisions.sum()
    print(f"\nCollision events detected: {collision_count} ({collision_count/len(state_df)*100:.2f}%)")
    
    # Get state bounds
    bounds = extractor.get_state_bounds(state_df)
    print(f"\nState bounds:")
    for key, (min_val, max_val) in bounds.items():
        print(f"  {key}: [{min_val:.2f}, {max_val:.2f}]")
    
    # Example: Get state vector at index 100
    if len(state_df) > 100:
        state_vec = extractor.get_state_vector(state_df, 100)
        print(f"\nState vector at index 100: {state_vec}")

