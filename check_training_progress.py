"""
Check training progress and find the latest model.
"""

import os
import glob
from pathlib import Path

def check_training_progress():
    """Check if training is complete and find the latest model."""
    
    models_dir = "./models_3d/"
    
    # Look for cruise_optimized model
    best_model_path = os.path.join(models_dir, "cruise_optimized_best", "best_model.zip")
    
    if os.path.exists(best_model_path):
        print(f"[OK] Best model found: {best_model_path}")
        
        # Check file size and modification time
        import time
        file_size = os.path.getsize(best_model_path) / (1024 * 1024)  # MB
        mod_time = os.path.getmtime(best_model_path)
        mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
        
        print(f"     File size: {file_size:.2f} MB")
        print(f"     Last modified: {mod_time_str}")
        
        return best_model_path
    else:
        print(f"[INFO] Best model not found yet: {best_model_path}")
        print(f"       Training may still be in progress...")
        
        # Check for any checkpoint files
        checkpoint_pattern = os.path.join(models_dir, "cruise_optimized_checkpoints", "*.zip")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if checkpoints:
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"[INFO] Latest checkpoint found: {latest_checkpoint}")
            return latest_checkpoint
        
        return None

if __name__ == "__main__":
    model_path = check_training_progress()
    if model_path:
        print(f"\n[SUCCESS] Model ready for analysis!")
        print(f"         Run: python analyze_helicopter_behavior.py --model_path {model_path} --n_episodes 20 --filter_phases")
    else:
        print(f"\n[INFO] Training still in progress. Check again later.")

