"""
Aşama 1 Modelini Engel Olmadan Test Et
"""

import os
import numpy as np
import argparse
from stable_baselines3 import PPO
from flight_env_3d import FlightControlEnv3D
from train_stage1_altitude_fixed import AltitudeFixedEnv
from train_stage1_target_reaching import TargetReachingEnv
from train_stage1_optimized import OptimizedTargetReachingEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_stage1_no_obstacle(
    model_path: str,
    num_episodes: int = 5
):
    """
    Aşama 1 modelini engel OLMADAN test et.
    """
    print("=" * 70)
    print("ASAMA 1 MODEL TESTI - ENGEL YOK")
    print("=" * 70)
    
    # Model yükle
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}")
        return
    
    print(f"[BILGI] Model yükleniyor: {model_path}")
    model = PPO.load(model_path)
    print("[BASARILI] Model yüklendi!")
    
    # Environment oluştur - ENGEL YOK! Optimize edilmiş versiyon
    # Önce OptimizedTargetReachingEnv'i dene
    try:
        env = OptimizedTargetReachingEnv(
            world_size=(1000.0, 1000.0, 300.0),  # Büyütülmüş dünya
            num_obstacles=0,
            max_episode_steps=5000,  # Uzun episode'lar
            render_mode=None,  # Görselleştirme kapalı (hızlı test için)
            use_log_data=True,
            log_data_path='fg_log2.csv',
            moving_obstacles=False,
            target_behind_obstacle=False,
            max_speed=15.0  # Azaltılmış hız
        )
    except:
        try:
            env = TargetReachingEnv(
                world_size=(500.0, 500.0, 200.0),
                num_obstacles=0,
                max_episode_steps=2000,
                render_mode='human',
                use_log_data=True,
                log_data_path='fg_log2.csv',
                moving_obstacles=False,
                target_behind_obstacle=False
            )
        except:
            env = AltitudeFixedEnv(
        world_size=(500.0, 500.0, 200.0),
        num_obstacles=0,  # ENGEL YOK!
        max_episode_steps=2000,
        render_mode='human',  # 3D görselleştirme açık
        use_log_data=True,
        log_data_path='fg_log2.csv',
        moving_obstacles=False,
        target_behind_obstacle=False  # Engel yok, hedef rastgele
    )
    
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'goal_reached_count': 0,
        'min_distances': []
    }
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        obs, info = env.reset(seed=episode * 42)
        episode_reward = 0
        episode_length = 0
        goal_reached = False
        min_distance = info['distance_to_target']
        
        print(f"Başlangıç pozisyonu: {info['position']}")
        print(f"Hedef pozisyon: {info['target']}")
        print(f"Başlangıç mesafesi: {info['distance_to_target']:.2f}m")
        print("\n[UÇUŞ BAŞLIYOR - ENGEL YOK, SADECE HEDEFE ULAŞMA...]")
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if info['distance_to_target'] < min_distance:
                min_distance = info['distance_to_target']
            
            # Hedefe ulaşıldı mı kontrol et (15m eşik - eğitimle uyumlu)
            # Debug: goal_reached durumunu kontrol et
            env_goal_reached = info.get('goal_reached', False)
            distance = info['distance_to_target']
            
            if env_goal_reached or distance < 15.0:
                if not goal_reached:
                    goal_reached = True
                    print(f"\n[BAŞARILI!] Hedefe ulaşıldı! Adım: {episode_length}, Mesafe: {distance:.2f}m")
                    print(f"[DEBUG] Environment goal_reached: {env_goal_reached}, Distance: {distance:.2f}m")
                    results['goal_reached_count'] += 1
                    results['success_count'] += 1
                    # Hedefe ulaşıldığında episode'u bitir
                    break
            
            # Debug: 15-20m aralığında mesafeyi göster
            if 15.0 <= distance < 20.0 and episode_length % 50 == 0:
                print(f"[YAKIN] Adım {episode_length}: Mesafe={distance:.2f}m (15m eşiğine {distance-15.0:.2f}m kaldı)")
            
            if episode_length % 100 == 0:
                print(f"Adım {episode_length}: Reward={episode_reward:.2f}, "
                      f"Mesafe={info['distance_to_target']:.2f}m")
            
            # env.render()  # Görselleştirme kapalı (hızlı test için)
            
            if terminated:
                if info.get('goal_reached', False):
                    print(f"\n[BAŞARILI!] Hedefe ulaşıldı ve episode bitti. Adım: {episode_length}")
                    if not goal_reached:
                        goal_reached = True
                        results['goal_reached_count'] += 1
                        results['success_count'] += 1
                else:
                    print(f"\n[ÇARPIŞMA!] Boundary collision. Adım: {episode_length}")
                break
            
            if truncated:
                print(f"\n[ZAMAN AŞIMI] Episode zaman aşımı. Adım: {episode_length}")
                if goal_reached:
                    results['success_count'] += 1
                break
        
        # Goal reached sayımı zaten yukarıda yapıldı, tekrar sayma
        
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        results['min_distances'].append(min_distance)
        
        print(f"\nEpisode Özeti:")
        print(f"  Toplam Reward: {episode_reward:.2f}")
        print(f"  Episode Uzunluğu: {episode_length} adım")
        print(f"  Hedefe Ulaşıldı: {'Evet' if goal_reached else 'Hayır'}")
        print(f"  Son Mesafe: {info['distance_to_target']:.2f}m")
        print(f"  Minimum Mesafe: {min_distance:.2f}m")
    
    # Genel sonuçlar
    print("\n" + "=" * 70)
    print("GENEL SONUÇLAR (ENGEL YOK)")
    print("=" * 70)
    print(f"Toplam Episode: {num_episodes}")
    print(f"Ortalama Reward: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    print(f"Ortalama Episode Uzunluğu: {np.mean(results['episode_lengths']):.1f} adım")
    print(f"Hedefe Ulaşma: {results['goal_reached_count']}/{num_episodes} ({results['goal_reached_count']/num_episodes*100:.1f}%)")
    print(f"Başarı Oranı: {results['success_count']/num_episodes*100:.1f}%")
    print(f"Ortalama Minimum Mesafe: {np.mean(results['min_distances']):.2f}m")
    
    env.close()
    print("\n[TEST TAMAMLANDI]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aşama 1 modelini engel olmadan test et')
    parser.add_argument('--model_path', type=str, 
                       default='./models_3d/curriculum_ppo_20260103_171922_stage1_best/best_model.zip',
                       help='Aşama 1 model yolu')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Test episode sayısı')
    
    args = parser.parse_args()
    
    test_stage1_no_obstacle(
        model_path=args.model_path,
        num_episodes=args.num_episodes
    )

