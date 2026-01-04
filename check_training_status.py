"""Eğitim durumunu kontrol et"""
import os
import glob
import re
from datetime import datetime

# En yeni stage1_optimized model klasörünü bul
model_dirs = [d for d in os.listdir('./models_3d') 
              if 'stage1_optimized' in d and os.path.isdir(f'./models_3d/{d}')]
model_dirs.sort(reverse=True)

if not model_dirs:
    print("[HATA] Henuz model klasoru olusturulmamis")
    exit(0)

latest_dir = model_dirs[0]
print(f"[BILGI] En yeni model: {latest_dir}")

# Checkpoint'leri kontrol et
checkpoint_dir = f'./models_3d/{latest_dir}_checkpoints'
if os.path.exists(checkpoint_dir):
    checkpoints = glob.glob(f'{checkpoint_dir}/*.zip')
    print(f"[BILGI] Checkpoint sayisi: {len(checkpoints)}")
    
    if checkpoints:
        # En yüksek step'i bul
        steps = []
        for cp in checkpoints:
            match = re.search(r'(\d+)_steps', cp)
            if match:
                steps.append(int(match.group(1)))
        
        if steps:
            max_step = max(steps)
            progress = (max_step / 500000) * 100
            print(f"[ILERLEME] En yuksek step: {max_step:,} / 500,000 ({progress:.1f}%)")
            
            # Son checkpoint zamanını bul
            latest_cp = max(checkpoints, key=os.path.getmtime)
            mod_time = os.path.getmtime(latest_cp)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[ZAMAN] Son checkpoint: {mod_time_str}")
        else:
            print("[UYARI] Checkpoint'lerde step bilgisi bulunamadi")
    else:
        print("[UYARI] Henuz checkpoint olusturulmamis")
else:
    print("[UYARI] Checkpoint klasoru henuz olusturulmamis")

# Best model kontrolü
best_dir = f'./models_3d/{latest_dir}_best'
if os.path.exists(best_dir):
    best_model = os.path.join(best_dir, 'best_model.zip')
    if os.path.exists(best_model):
        mod_time = os.path.getmtime(best_model)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[BEST MODEL] Olusturulma zamani: {mod_time_str}")
    else:
        print("[UYARI] Best model henuz olusturulmamis")
else:
    print("[UYARI] Best model klasoru henuz olusturulmamis")

