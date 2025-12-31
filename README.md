# 🚁 3D Helicopter Flight Control with Reinforcement Learning

Bu proje, Reinforcement Learning (RL) kullanarak 3D ortamda helikopter kontrolü öğrenen bir sistemdir. Helikopter, engellerden kaçınarak hedefe ulaşmayı öğrenir. Sistem, gerçek FlightGear simülasyon verilerini kullanarak gerçekçi helikopter davranışı sağlar.

---

## 🛠️ Kurulum

### Gereksinimler

- Python 3.8+
- CUDA (GPU için, opsiyonel)

### Adım 1: Bağımlılıkları Yükle

pip install -r requirements.txt
### Adım 2: Veri Dosyasını Kontrol Et

`fg_log2.csv` dosyasının proje dizininde olduğundan emin olun. Bu dosya FlightGear telemetry verilerini içerir ve gerçekçi helikopter davranışı için kullanılır.

---

## 🚀 Hızlı Başlangıç

### 1. Environment'i Test Et

python flight_env_3d.pyBu komut environment'i test eder ve 50 rastgele adım çalıştırır. 3D görselleştirme açılır.

### 2. Model Eğit

python train_3d_ppo.py --total_timesteps 200000 --model_name my_helicopter_model### 3. Modeli Test Et

python test_3d_environment.py \
    --model_path ./models_3d/my_helicopter_model_best/best_model.zip \
    --num_episodes 5---

## 🎮 Environment Nasıl Çalıştırılır

### Manuel Test (Kod ile)

from flight_env_3d import FlightControlEnv3D
import numpy as np

# Environment oluştur
env = FlightControlEnv3D(
    world_size=(500.0, 500.0, 200.0),  # Dünya boyutları (x, y, z) metre
    num_obstacles=1,                     # Engel sayısı
    max_episode_steps=2000,              # Maksimum adım sayısı
    render_mode='human',                 # Görselleştirme: 'human' veya None
    use_log_data=True,                   # FlightGear log verilerini kullan
    log_data_path='fg_log2.csv',         # Log dosyası yolu
    moving_obstacles=False,              # Hareketli engeller (False = sabit)
    target_behind_obstacle=True         # Target engelin arkasında
)

# Environment'i sıfırla
obs, info = env.reset(seed=42)

print(f"Initial position: {info['position']}")
print(f"Target position: {info['target']}")
print(f"Distance to target: {info['distance_to_target']:.2f}m")

# Rastgele aksiyonlar ile test et
for step in range(100):
    action = env.action_space.sample()  # Rastgele aksiyon
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: Reward={reward:.2f}, Distance={info['distance_to_target']:.2f}m")
    
    if terminated:
        print("Collision detected!")
        break
    if truncated:
        print("Timeout!")
        break
    
    # Görselleştir
    env.render()

env.close()### Environment Parametreleri

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `world_size` | Tuple | (500, 500, 200) | 3D dünya boyutları (x, y, z) metre |
| `num_obstacles` | int | 1 | Engel sayısı (şu an sadece 1 sabit engel) |
| `max_episode_steps` | int | 2000 | Maksimum adım sayısı |
| `render_mode` | str | None | 'human' = görselleştirme, None = görselleştirme yok |
| `use_log_data` | bool | False | FlightGear log verilerini kullan |
| `log_data_path` | str | 'fg_log2.csv' | Log dosyası yolu |
| `moving_obstacles` | bool | False | Hareketli engeller (şu an False) |
| `target_behind_obstacle` | bool | True | Target engelin arkasında |

### State Space (Gözlem)

Environment 10 boyutlu state vektörü döndürür:
on
state = [x, y, z, vx, vy, vz, roll, pitch, yaw, altitude_rate]
#        |  position  |  velocity  |    attitude    |  rate- **Position**: `[x, y, z]` - Helikopter pozisyonu (metre)
- **Velocity**: `[vx, vy, vz]` - Hız vektörü (m/s)
- **Attitude**: `[roll, pitch, yaw]` - Açılar (derece)
- **Altitude Rate**: `altitude_rate` - İrtifa değişim hızı (m/s)

### Action Space (Aksiyon)

Environment 4 boyutlu aksiyon vektörü kabul eder:

action = [roll_command, pitch_command, yaw_command, throttle_command]
#         |              attitude commands              |  throttle- **Roll Command**: Yatış komutu (-1.0 ile 1.0 arası)
- **Pitch Command**: Yükseliş/dalış komutu (-1.0 ile 1.0 arası)
- **Yaw Command**: Dönüş komutu (-1.0 ile 1.0 arası)
- **Throttle Command**: Gaz komutu (-1.0 ile 1.0 arası)

---

## 🎓 Model Eğitimi ve Kaydetme

### Temel Eğitim

python train_3d_ppo.py --total_timesteps 200000 --model_name my_model### Gelişmiş Eğitim (Daha Fazla Parametre)

python train_3d_ppo.py \
    --total_timesteps 500000 \
    --learning_rate 3e-4 \
    --num_obstacles 1 \
    --model_name advanced_model \
    --log_dir ./logs_3d/ \
    --save_dir ./models_3d/### Eğitim Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--total_timesteps` | 200000 | Toplam eğitim adımı |
| `--learning_rate` | 3e-4 | Öğrenme oranı |
| `--num_obstacles` | 1 | Engel sayısı |
| `--model_name` | Otomatik | Model adı (tarih-saat ile) |
| `--log_dir` | ./logs_3d/ | TensorBoard log dizini |
| `--save_dir` | ./models_3d/ | Model kayıt dizini |

### Model Kaydetme

Eğitim sırasında otomatik olarak 3 tip model kaydedilir:

1. **Best Model** (En İyi Model)
   - Yolu: `./models_3d/{model_name}_best/best_model.zip`
   - Açıklama: Evaluation sırasında en yüksek reward'a sahip model
   - **Kullanım**: Test için en iyi seçenek

2. **Checkpoints** (Ara Kayıtlar)
   - Yolu: `./models_3d/{model_name}_checkpoints/{model_name}_50000_steps.zip`
   - Açıklama: Belirli adım sayılarında kaydedilen modeller
   - Kullanım: Eğitimi yarıda kesip devam ettirmek için

3. **Final Model** (Final Model)
   - Yolu: `./models_3d/{model_name}_final.zip`
   - Açıklama: Eğitimin sonunda kaydedilen model
   - Kullanım: Son durumu saklamak için


python test_3d_environment.py \
    --model_path ./models_3d/my_helicopter_model_best/best_model.zip \
    --num_episodes 5z (★)**: Target
- **Kırmızı Silindir**: Engel
- **Mavi Çizgi**: Helikopter trajectory'si

---

## 📁 Kod Dosyaları ve Açıklamaları

### 1. `flight_env_3d.py` - 3D Environment

**Ne İşe Yarar:**
- 3D helikopter uçuş ortamını tanımlar
- Gymnasium-compatible environment sağlar
- Engel kaçınma, collision detection, reward hesaplama yapar

**Ana Sınıflar:**
- `Obstacle3D`: 3D engel tanımı (silindir/küre)
- `FlightControlEnv3D`: Ana environment sınıfı

**Önemli Metodlar:**
- `reset()`: Environment'i sıfırla, yeni episode başlat
- `step(action)`: Bir adım ilerle, reward döndür
- `render()`: 3D görselleştirme
- `_check_collisions()`: Çarpışma kontrolü
- `_generate_obstacles()`: Engel oluşturma

**Kullanım:**
python
from flight_env_3d import FlightControlEnv3D
env = FlightControlEnv3D(render_mode='human')---

### 2. `train_3d_ppo.py` - Model Eğitimi

**Ne İşe Yarar:**
- PPO (Proximal Policy Optimization) algoritması ile model eğitir
- Stable-Baselines3 kullanır
- Otomatik model kaydetme ve evaluation yapar

**Ana Fonksiyonlar:**
- `train_ppo_3d()`: Eğitim fonksiyonu
- `make_env_3d()`: Environment factory

**Özellikler:**
- Otomatik best model kaydetme
- Checkpoint kaydetme (belirli adımlarda)
- TensorBoard logging
- Evaluation callback

**Kullanım:**
python train_3d_ppo.py --total_timesteps 200000 --model_name my_model---

### 3. `test_3d_environment.py` - Model Testi

**Ne İşe Yarar:**
- Eğitilmiş modeli test eder
- Senaryoları çalıştırır
- Performans metrikleri toplar

**Ana Fonksiyon:**
- `test_model_3d()`: Test fonksiyonu

**Toplanan Metrikler:**
- Episode rewards
- Episode lengths
- Success rate
- Goal reached count
- Collision count

**Kullanım:**
python test_3d_environment.py --model_path ./models_3d/model_best/best_model.zip---

### 4. `state_extractor.py` - State Extraction

**Ne İşe Yarar:**
- FlightGear log verilerinden state vektörü çıkarır
- Collision detection yapar
- State bounds hesaplar

**Ana Sınıf:**
- `StateExtractor`: State extraction sınıfı

**Çıkardığı State:**hon
state = [altitude_agl, roll, pitch, heading, altitude_rate]**Metodlar:**
- `load_data()`: CSV dosyasını yükle
- `extract_states()`: State vektörlerini çıkar
- `detect_collisions()`: Çarpışma tespiti
- `get_state_bounds()`: State sınırları

**Kullanım:**n
from state_extractor import StateExtractor
extractor = StateExtractor()
df = extractor.load_data('fg_log2.csv')
states = extractor.extract_states(df)---

### 5. `reward_design.py` - Reward Function

**Ne İşe Yarar:**
- Reward function tasarımı ve hesaplama
- Reward bileşenlerini analiz eder
- Görselleştirme sağlar

**Ana Sınıf:**
- `RewardDesigner`: Reward tasarım sınıfı

**Reward Bileşenleri:**
reward = R_altitude + R_stability + R_collision + R_goal + R_progress- **R_altitude**: Güvenli irtifa için ödül
- **R_stability**: Roll/pitch stabilitesi için ceza
- **R_collision**: Çarpışma için büyük ceza
- **R_goal**: Hedefe ulaşma için ödül
- **R_progress**: Hedefe yaklaşma için ödül

**Metodlar:**
- `compute_reward()`: Reward hesapla
- `compute_batch_rewards()`: Toplu reward hesaplama
- `visualize_rewards()`: Reward görselleştirme

**Kullanım:**
from reward_design import RewardDesigner
designer = RewardDesigner()
reward = designer.compute_reward(state, collision=False)---

### 6. `analyze_helicopter_behavior.py` - Behavior Analysis

**Ne İşe Yarar:**
- Eğitilmiş modelin davranışını analiz eder
- Gerçek helikopter log verileriyle karşılaştırır
- Similarity score hesaplar

**Ana Sınıf:**
- `HelicopterBehaviorAnalyzer`: Analiz sınıfı

**Analiz Metrikleri:**
- Roll/Pitch angle similarity
- Altitude rate similarity
- Speed similarity
- Angular rates similarity
- Overall similarity score

**Çıktılar:**
- `behavior_comparison.png`: Görselleştirme
- `behavior_report.txt`: Detaylı rapor

**Kullanım:**
python analyze_helicopter_behavior.py \
    --model_path ./models_3d/model_best/best_model.zip \
    --n_episodes 20---

### 7. `improve_with_log_data.py` - Log Data Analysis

**Ne İşe Yarar:**
- Log verisi kalitesini analiz eder
- İyileştirme önerileri sunar
- Enhanced environment config önerir

**Ana Sınıf:**
- `LogDataEnhancer`: Log verisi analiz sınıfı

**Analiz Edilenler:**
- Veri hacmi ve çeşitlilik
- Uçuş fazları (climbing, level, descending)
- Maneuver çeşitliliği
- Kalite önerileri

**Kullanım:**
python improve_with_log_data.py---

## 📊 Analiz Araçları

### 1. Behavior Analysis

Modelin gerçek helikopter gibi davranıp davranmadığını kontrol edin:

python analyze_helicopter_behavior.py \
    --model_path ./models_3d/my_model_best/best_model.zip \
    --n_episodes 20 \
    --output_dir ./behavior_analysis/**Çıktılar:**
- Similarity scores (0-100%)
- Görselleştirmeler
- Detaylı rapor

### 2. Log Data Quality

Log verisi kalitesini analiz edin:

python improve_with_log_data.py**Çıktılar:**
- Veri kalitesi analizi
- İyileştirme önerileri
- Enhanced config önerileri

---

## ❓ Sık Sorulan Sorular

### Q: Model nerede kaydediliyor?

A: Modeller `./models_3d/` dizininde kaydedilir:
- **Best model**: `./models_3d/{model_name}_best/best_model.zip`
- **Checkpoints**: `./models_3d/{model_name}_checkpoints/`
- **Final**: `./models_3d/{model_name}_final.zip`

### Q: Eğitimi yarıda kesip devam ettirebilir miyim?

A: Evet! Checkpoint'lerden devam edebilirsiniz:

from stable_baselines3 import PPO

# Checkpoint'ten yükle
model = PPO.load('./models_3d/my_model_checkpoints/my_model_200000_steps.zip')

# Eğitime devam et
model.learn(total_timesteps=300000)  # Toplam 500k olacak### Q: Görselleştirme nasıl kapatılır?

A: `render_mode=None` kullanın veya test scriptinde `--render` parametresini kullanmayın.

### Q: Farklı engel sayıları ile eğitebilir miyim?

A: Şu an sadece 1 sabit engel destekleniyor. Kodda `num_obstacles=1` olarak ayarlanmış.

### Q: Log verileri nasıl kullanılıyor?

A: Log verileri başlangıç durumları için kullanılıyor:
- Initial attitude (roll, pitch, heading)
- Initial velocity
- Gerçekçi helikopter davranışı sağlar

### Q: Similarity score ne anlama geliyor?

A: Similarity score (0-100%), modelin gerçek helikopter davranışına ne kadar benzediğini gösterir:
- **>70%**: İyi benzerlik ✅
- **50-70%**: Orta benzerlik ⚠️
- **<50%**: Düşük benzerlik ❌

### Q: TensorBoard'da ne görürüm?

A: TensorBoard'da şunları görürsünüz:
- Episode reward
- Episode length
- Learning rate
- Policy loss
- Value loss
- Evaluation metrics

---

## 📚 Ek Kaynaklar

- **3D Environment Guide**: `3D_ENVIRONMENT_GUIDE.md`
- **Training Guide**: `TRAINING_GUIDE.md`
- **Behavior Analysis Guide**: `BEHAVIOR_ANALYSIS_GUIDE.md`

---

## 🐛 Sorun Giderme

### Problem: "Model not found"

**Çözüm:** Model yolunu kontrol edin:
ls ./models_3d/my_model_best/### Problem: "Import error"

**Çözüm:** Bağımlılıkları yükleyin:
pip install -r requirements.txt### Problem: Görselleştirme açılmıyor

**Çözüm:** `render_mode='human'` kullandığınızdan emin olun ve matplotlib backend'inin doğru olduğunu kontrol edin.

### Problem: Eğitim çok yavaş

**Çözüm:** 
- GPU kullanın (CUDA)
- `total_timesteps` değerini azaltın
- `n_steps` parametresini ayarlayın

---

