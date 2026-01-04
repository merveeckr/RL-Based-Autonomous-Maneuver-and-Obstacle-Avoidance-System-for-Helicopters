"""Basit eğitim durumu kontrolü"""
import os
import glob
from datetime import datetime

# En yeni PPO log klasörünü bul
ppo_dirs = [d for d in os.listdir('./logs_3d') if d.startswith('PPO_')]
if ppo_dirs:
    ppo_dirs.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0, reverse=True)
    latest_ppo = ppo_dirs[0]
    print(f"[BILGI] En yeni TensorBoard log: {latest_ppo}")
    
    log_files = glob.glob(f'./logs_3d/{latest_ppo}/events.out.*')
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        mod_time = os.path.getmtime(latest_log)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ZAMAN] Son log guncellemesi: {mod_time_str}")
        
        # Dosya boyutu
        size = os.path.getsize(latest_log) / 1024  # KB
        print(f"[BOYUT] Log dosyasi: {size:.1f} KB")
    else:
        print("[UYARI] Log dosyasi bulunamadi")

# En yeni model klasörünü kontrol et
model_dirs = [d for d in os.listdir('./models_3d') 
              if 'stage1_optimized' in d and not '_checkpoints' in d and not '_best' in d and not '_eval' in d]
if model_dirs:
    model_dirs.sort(reverse=True)
    latest_model = model_dirs[0]
    print(f"\n[BILGI] En yeni model klasoru: {latest_model}")
    
    # Checkpoint kontrolü
    checkpoint_dir = f'./models_3d/{latest_model}_checkpoints'
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(f'{checkpoint_dir}/*.zip')
        print(f"[BILGI] Checkpoint sayisi: {len(checkpoints)}")
        if checkpoints:
            latest_cp = max(checkpoints, key=os.path.getmtime)
            mod_time = os.path.getmtime(latest_cp)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[ZAMAN] Son checkpoint: {mod_time_str}")
    else:
        print("[UYARI] Checkpoint klasoru henuz olusturulmamis")

print("\n[NOT] Eğitim devam ediyorsa, TensorBoard ile izleyebilirsiniz:")
print("      tensorboard --logdir ./logs_3d/")

