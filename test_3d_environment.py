"""
Test 3D Environment with Trained Model
Tests a trained PPO model in the 3D obstacle avoidance environment.
"""

import os
import numpy as np
import argparse
from stable_baselines3 import PPO
from flight_env_3d import FlightControlEnv3D


def test_model_3d(
    model_path: str,
    num_episodes: int = 5,
    render: bool = True,
    world_size: tuple = (500.0, 500.0, 200.0),
    num_obstacles: int = 5
):
    """
    Test trained model in 3D environment.
    
    Args:
        model_path: Path to trained PPO model
        num_episodes: Number of test episodes
        render: Whether to render the environment
        world_size: Size of 3D world
        num_obstacles: Number of obstacles
    """
    print("=" * 70)
    print("3D ENVIRONMENT TEST")
    print("=" * 70)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    print(f"[INFO] Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("[SUCCESS] Model loaded!")
    
    # Create environment
    env = FlightControlEnv3D(
        world_size=world_size,
        num_obstacles=1,  # Single fixed obstacle
        max_episode_steps=2000,
        render_mode='human' if render else None,
        use_log_data=True,  # Use FlightGear log data for realistic helicopter behavior
        log_data_path='fg_log2.csv',
        moving_obstacles=False,  # Fixed obstacles
        obstacle_speed=0.0,  # No movement
        target_behind_obstacle=True  # Place target behind obstacle
    )
    
    # Test episodes
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'collision_count': 0,
        'goal_reached_count': 0
    }
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        obs, info = env.reset(seed=episode * 42)
        episode_reward = 0
        episode_length = 0
        goal_reached = False
        
        print(f"Initial position: {info['position']}")
        print(f"Target position: {info['target']}")
        print(f"Initial distance: {info['distance_to_target']:.2f}m")
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Check if goal reached
            if info['distance_to_target'] < 10.0 and not goal_reached:
                goal_reached = True
                print(f"[SUCCESS] Goal reached at step {episode_length}!")
            
            # Render
            if render:
                env.render()
            
            # Print progress
            if episode_length % 100 == 0:
                print(f"Step {episode_length}: Reward={episode_reward:.2f}, "
                      f"Distance={info['distance_to_target']:.2f}m, "
                      f"Min obstacle dist={info['min_obstacle_distance']:.2f}m")
            
            if terminated:
                print(f"[COLLISION] Episode ended due to collision at step {episode_length}")
                results['collision_count'] += 1
                break
            
            if truncated:
                print(f"[TIMEOUT] Episode ended due to timeout at step {episode_length}")
                if goal_reached:
                    results['success_count'] += 1
                break
        
        if goal_reached:
            results['goal_reached_count'] += 1
        
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        
        print(f"\nEpisode Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Goal Reached: {goal_reached}")
        print(f"  Final Distance: {info['distance_to_target']:.2f}m")
    
    # Print overall results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    print(f"Mean Episode Length: {np.mean(results['episode_lengths']):.1f}")
    print(f"Goal Reached: {results['goal_reached_count']}/{num_episodes} ({results['goal_reached_count']/num_episodes*100:.1f}%)")
    print(f"Collisions: {results['collision_count']}/{num_episodes} ({results['collision_count']/num_episodes*100:.1f}%)")
    print(f"Success Rate: {results['success_count']/num_episodes*100:.1f}%")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained model in 3D environment')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PPO model')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of test episodes')
    parser.add_argument('--no_render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--num_obstacles', type=int, default=5,
                       help='Number of obstacles')
    
    args = parser.parse_args()
    
    test_model_3d(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        render=not args.no_render,
        num_obstacles=args.num_obstacles
    )

