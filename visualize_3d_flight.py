"""
3D Flight Visualization Script
Görselleştirme ile model testi yapar ve sonuçları gösterir.
"""

import os
import numpy as np
import argparse
from stable_baselines3 import PPO
from flight_env_3d import FlightControlEnv3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_model_3d(
    model_path: str,
    num_episodes: int = 3,
    world_size: tuple = (500.0, 500.0, 200.0),
    save_trajectories: bool = True
):
    """
    Modeli test et ve 3D görselleştirme ile göster.
    """
    print("=" * 70)
    print("3D HELİKOPTER UÇUŞ GÖRSELLEŞTİRMESİ")
    print("=" * 70)
    
    # Model yükle
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}")
        return
    
    print(f"[BİLGİ] Model yükleniyor: {model_path}")
    model = PPO.load(model_path)
    print("[BAŞARILI] Model yüklendi!")
    
    # Environment oluştur
    env = FlightControlEnv3D(
        world_size=world_size,
        num_obstacles=1,
        max_episode_steps=2000,
        render_mode='human',  # 3D görselleştirme açık
        use_log_data=True,
        log_data_path='fg_log2.csv',
        moving_obstacles=False,
        obstacle_speed=0.0,
        target_behind_obstacle=True
    )
    
    # Sonuçlar
    all_trajectories = []
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'collision_count': 0,
        'goal_reached_count': 0,
        'min_distances': []
    }
    
    # Test episode'ları
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"EPİSODE {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        obs, info = env.reset(seed=episode * 42)
        episode_reward = 0
        episode_length = 0
        goal_reached = False
        trajectory = [info['position'].copy()]
        min_distance = info['distance_to_target']
        # min_obstacle_distance reset'te olmayabilir, step'te olacak
        min_obstacle_dist = info.get('min_obstacle_distance', float('inf'))
        
        print(f"Başlangıç pozisyonu: {info['position']}")
        print(f"Hedef pozisyon: {info['target']}")
        print(f"Başlangıç mesafesi: {info['distance_to_target']:.2f}m")
        # Reset'te min_obstacle_distance yok, step'te olacak
        initial_obstacle_dist = env._get_min_obstacle_distance(info['position'])
        print(f"Engel mesafesi: {initial_obstacle_dist:.2f}m")
        print("\n[UÇUŞ BAŞLIYOR - 3D görselleştirme açık...]")
        
        step_count = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            trajectory.append(info['position'].copy())
            
            # Minimum mesafeleri güncelle
            if info['distance_to_target'] < min_distance:
                min_distance = info['distance_to_target']
            if info['min_obstacle_distance'] < min_obstacle_dist:
                min_obstacle_dist = info['min_obstacle_distance']
            
            # Hedefe ulaşıldı mı?
            if info['distance_to_target'] < 10.0 and not goal_reached:
                goal_reached = True
                print(f"\n[BAŞARILI!] Hedefe ulaşıldı! Adım: {episode_length}")
            
            # Her 50 adımda bilgi ver
            if step_count % 50 == 0:
                print(f"Adım {step_count}: Reward={episode_reward:.2f}, "
                      f"Mesafe={info['distance_to_target']:.2f}m, "
                      f"Engel mesafesi={info['min_obstacle_distance']:.2f}m")
            
            # Render (3D görselleştirme)
            env.render()
            
            if terminated:
                print(f"\n[ÇARPIŞMA!] Episode çarpışma ile sonlandı. Adım: {episode_length}")
                results['collision_count'] += 1
                break
            
            if truncated:
                print(f"\n[ZAMAN AŞIMI] Episode zaman aşımı ile sonlandı. Adım: {episode_length}")
                if goal_reached:
                    results['success_count'] += 1
                break
        
        if goal_reached:
            results['goal_reached_count'] += 1
        
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        results['min_distances'].append(min_distance)
        all_trajectories.append(np.array(trajectory))
        
        print(f"\nEpisode Özeti:")
        print(f"  Toplam Reward: {episode_reward:.2f}")
        print(f"  Episode Uzunluğu: {episode_length} adım")
        print(f"  Hedefe Ulaşıldı: {'Evet' if goal_reached else 'Hayır'}")
        print(f"  Son Mesafe: {info['distance_to_target']:.2f}m")
        print(f"  Minimum Mesafe: {min_distance:.2f}m")
        print(f"  Minimum Engel Mesafesi: {min_obstacle_dist:.2f}m")
    
    # Genel sonuçlar
    print("\n" + "=" * 70)
    print("GENEL SONUÇLAR")
    print("=" * 70)
    print(f"Toplam Episode: {num_episodes}")
    print(f"Ortalama Reward: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    print(f"Ortalama Episode Uzunluğu: {np.mean(results['episode_lengths']):.1f} adım")
    print(f"Hedefe Ulaşma: {results['goal_reached_count']}/{num_episodes} ({results['goal_reached_count']/num_episodes*100:.1f}%)")
    print(f"Çarpışma: {results['collision_count']}/{num_episodes} ({results['collision_count']/num_episodes*100:.1f}%)")
    print(f"Başarı Oranı: {results['success_count']/num_episodes*100:.1f}%")
    print(f"Ortalama Minimum Mesafe: {np.mean(results['min_distances']):.2f}m")
    
    # Trajectory'leri kaydet
    if save_trajectories and all_trajectories:
        print(f"\n[KAYDEDİLİYOR] Trajectory'ler kaydediliyor...")
        save_dir = "./flight_visualizations/"
        os.makedirs(save_dir, exist_ok=True)
        
        # Tüm trajectory'leri bir figürde göster
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Her episode için trajectory çiz
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, traj in enumerate(all_trajectories):
            color = colors[i % len(colors)]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=color, linewidth=2, alpha=0.7, 
                   label=f'Episode {i+1}')
            # Başlangıç noktası
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                      color=color, marker='o', s=100, 
                      label=f'Başlangıç {i+1}')
        
        # Son episode'dan engel ve hedef bilgilerini al
        env.reset(seed=0)
        # Engel çiz
        for obstacle in env.obstacles:
            if obstacle.type == 'cylinder':
                z_bottom = obstacle.position[2]
                z_top = obstacle.position[2] + obstacle.height
                theta = np.linspace(0, 2*np.pi, 20)
                x_circle = obstacle.position[0] + obstacle.radius * np.cos(theta)
                y_circle = obstacle.position[1] + obstacle.radius * np.sin(theta)
                ax.plot(x_circle, y_circle, [z_bottom]*len(theta), 'r-', linewidth=3, label='Engel')
                ax.plot(x_circle, y_circle, [z_top]*len(theta), 'r-', linewidth=3)
        
        # Hedef çiz
        ax.scatter([env.target_position[0]], [env.target_position[1]], [env.target_position[2]],
                  c='gold', marker='*', s=500, label='Hedef', edgecolors='black', linewidths=2)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D Helikopter Uçuş Trajectory\'leri', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # Eksen limitleri
        ax.set_xlim([-world_size[0]/2, world_size[0]/2])
        ax.set_ylim([-world_size[1]/2, world_size[1]/2])
        ax.set_zlim([0, world_size[2]])
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "flight_trajectories.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[KAYDEDİLDİ] Görselleştirme kaydedildi: {save_path}")
        plt.close()
    
    env.close()
    print("\n[TEST TAMAMLANDI]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D helikopter uçuş görselleştirmesi')
    parser.add_argument('--model_path', type=str, 
                       default='./models_3d/cruise_nav_obs_aware_v2_best/best_model.zip',
                       help='Eğitilmiş model yolu')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Test episode sayısı')
    parser.add_argument('--no_save', action='store_true',
                       help='Trajectory kaydetme')
    
    args = parser.parse_args()
    
    visualize_model_3d(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        save_trajectories=not args.no_save
    )

