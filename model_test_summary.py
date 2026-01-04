"""
Model Test Özet Raporu
Tüm modelleri test edip sonuçları karşılaştırır.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from flight_env_3d import FlightControlEnv3D


def test_model_quick(model_path, num_episodes=3):
    """Hızlı model testi - sadece uyumlu modeller için."""
    try:
        model = PPO.load(model_path)
        
        # Model'in observation space'ini kontrol et
        model_obs_dim = model.observation_space.shape[0]
        print(f"  Model observation space: {model_obs_dim}D")
        
        # Environment oluştur
        env = FlightControlEnv3D(
            world_size=(500.0, 500.0, 200.0),
            num_obstacles=1,
            max_episode_steps=2000,
            render_mode=None,  # Görselleştirme kapalı (hızlı test)
            use_log_data=True,
            log_data_path='fg_log2.csv',
            moving_obstacles=False,
            target_behind_obstacle=True
        )
        
        env_obs_dim = env.observation_space.shape[0]
        print(f"  Environment observation space: {env_obs_dim}D")
        
        # Uyumsuzluk kontrolü
        if model_obs_dim != env_obs_dim:
            print(f"  [UYUMSUZLUK] Model {model_obs_dim}D bekliyor, environment {env_obs_dim}D veriyor!")
            return None
        
        # Test
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0,
            'collision_count': 0,
            'goal_reached_count': 0,
            'min_distances': []
        }
        
        for episode in range(num_episodes):
            obs, info = env.reset(seed=episode * 42)
            episode_reward = 0
            episode_length = 0
            goal_reached = False
            min_distance = info['distance_to_target']
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if info['distance_to_target'] < min_distance:
                    min_distance = info['distance_to_target']
                
                if info['distance_to_target'] < 10.0 and not goal_reached:
                    goal_reached = True
                    results['goal_reached_count'] += 1
                
                if terminated:
                    results['collision_count'] += 1
                    break
                
                if truncated:
                    if goal_reached:
                        results['success_count'] += 1
                    break
            
            results['episode_rewards'].append(episode_reward)
            results['episode_lengths'].append(episode_length)
            results['min_distances'].append(min_distance)
        
        env.close()
        
        return {
            'mean_reward': np.mean(results['episode_rewards']),
            'std_reward': np.std(results['episode_rewards']),
            'mean_length': np.mean(results['episode_lengths']),
            'success_rate': results['success_count'] / num_episodes * 100,
            'collision_rate': results['collision_count'] / num_episodes * 100,
            'goal_reached_rate': results['goal_reached_count'] / num_episodes * 100,
            'mean_min_distance': np.mean(results['min_distances'])
        }
    except Exception as e:
        print(f"  [HATA] {str(e)}")
        return None


def main():
    """Tüm modelleri test et."""
    print("=" * 70)
    print("MODEL TEST ÖZET RAPORU")
    print("=" * 70)
    
    models_dir = "./models_3d/"
    model_paths = [
        ("cruise_nav_aggressive_v3", "cruise_nav_aggressive_v3_best/best_model.zip"),
        ("cruise_nav_obs_aware_v2", "cruise_nav_obs_aware_v2_best/best_model.zip"),
        ("cruise_optimized", "cruise_optimized_best/best_model.zip"),
        ("ppo_3d_20251231_150717", "ppo_3d_20251231_150717_best/best_model.zip"),
    ]
    
    results_summary = []
    
    for model_name, model_file in model_paths:
        model_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"\n[{model_name}] Model bulunamadı: {model_path}")
            continue
        
        print(f"\n[{model_name}] Test ediliyor...")
        print(f"  Yol: {model_path}")
        
        result = test_model_quick(model_path, num_episodes=3)
        
        if result:
            results_summary.append((model_name, result))
            print(f"  Ortalama Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"  Ortalama Episode Uzunluğu: {result['mean_length']:.1f} adım")
            print(f"  Başarı Oranı: {result['success_rate']:.1f}%")
            print(f"  Çarpışma Oranı: {result['collision_rate']:.1f}%")
            print(f"  Hedefe Ulaşma: {result['goal_reached_rate']:.1f}%")
            print(f"  Ortalama Minimum Mesafe: {result['mean_min_distance']:.2f}m")
        else:
            print(f"  [UYUMSUZ] Bu model test edilemedi (observation space uyumsuzluğu)")
    
    # Özet
    print("\n" + "=" * 70)
    print("ÖZET KARŞILAŞTIRMA")
    print("=" * 70)
    
    if results_summary:
        print(f"{'Model':<30} {'Reward':<15} {'Başarı':<10} {'Çarpışma':<10} {'Min Mesafe':<12}")
        print("-" * 70)
        for model_name, result in results_summary:
            print(f"{model_name:<30} {result['mean_reward']:>8.2f} ± {result['std_reward']:<5.2f} "
                  f"{result['success_rate']:>6.1f}%   {result['collision_rate']:>6.1f}%   "
                  f"{result['mean_min_distance']:>8.2f}m")
        
        # En iyi model
        best_model = max(results_summary, key=lambda x: x[1]['mean_reward'])
        print(f"\n[EN İYİ MODEL] {best_model[0]}")
        print(f"  Reward: {best_model[1]['mean_reward']:.2f}")
        print(f"  Başarı Oranı: {best_model[1]['success_rate']:.1f}%")
    else:
        print("[UYARI] Hiçbir model test edilemedi!")
        print("\n[ÖNERİ] Yeni bir model eğitmeniz gerekiyor.")
        print("        Environment şu an 18D observation space kullanıyor,")
        print("        ama eski modeller 10D bekliyor.")


if __name__ == "__main__":
    main()

