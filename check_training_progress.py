"""Eğitim ilerlemesini kontrol et"""
import os
import glob
import re
from datetime import datetime

# Checkpoint kontrolü
checkpoint_dir = './models_3d/stage1_optimized_20260104_043539_checkpoints'
if os.path.exists(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.zip'))
    if checkpoints:
        steps = []
        for f in checkpoints:
            match = re.search(r'(\d+)_steps', os.path.basename(f))
            if match:
                steps.append(int(match.group(1)))
        
        if steps:
            max_step = max(steps)
            latest = [f for f in checkpoints if f'{max_step}_steps' in f][0]
            mtime = os.path.getmtime(latest)
            age = datetime.now().timestamp() - mtime
            
            print(f"[CHECKPOINT] Son checkpoint: {max_step:,} / 1,500,000 step ({max_step/1500000*100:.1f}%)")
            print(f"[KALAN] {1500000-max_step:,} step kaldi")
            print(f"[DOSYA] {os.path.basename(latest)}")
            print(f"[GUNCELLEME] {int(age)} saniye once")
        else:
            print("[CHECKPOINT] Checkpoint bulunamadi")
    else:
        print("[CHECKPOINT] Henuz checkpoint yok")
else:
    print("[CHECKPOINT] Checkpoint dizini bulunamadi")

# Log kontrolü
log_dir = './logs/stage1_optimized_20260104_043539'
if os.path.exists(log_dir):
    logs = glob.glob(os.path.join(log_dir, '*.csv'))
    if logs:
        latest_log = max(logs, key=os.path.getmtime)
        mtime = os.path.getmtime(latest_log)
        age = datetime.now().timestamp() - mtime
        
        print(f"\n[LOG] Son log guncellemesi: {int(age)} saniye once")
        print(f"[DOSYA] {os.path.basename(latest_log)}")
        
        # Son satırı oku
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Header + en az 1 data satırı
                    last_line = lines[-1].strip()
                    print(f"[SON SATIR] {last_line[:100]}...")
        except Exception as e:
            print(f"[HATA] Log okunamadi: {e}")
    else:
        print("\n[LOG] Henuz log dosyasi yok")
else:
    print("\n[LOG] Log dizini bulunamadi")
