# 🚀 Model Eğitimi ve Test Senaryoları Kılavuzu

Bu kılavuz, PPO modeli eğitimi, hyperparameter tuning ve test senaryoları için adım adım talimatlar içerir.

## 📋 İçindekiler

1. [Hızlı Başlangıç](#hızlı-başlangıç)
2. [Model Eğitimi](#model-eğitimi)
3. [Test Senaryoları](#test-senaryoları)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Model Karşılaştırma](#model-karşılaştırma)

---

## 🎯 Hızlı Başlangıç

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. İlk Modeli Eğit

```bash
python train_ppo.py --total_timesteps 100000
```

Bu komut:
- 100,000 timestep eğitim yapar
- Modeli `./models/` klasörüne kaydeder
- TensorBoard loglarını `./logs/` klasörüne yazar

---

## 🏋️ Model Eğitimi

### Temel Kullanım

```bash
python train_ppo.py --total_timesteps 200000 --learning_rate 3e-4
```

### Tüm Parametreler

```bash
python train_ppo.py \
    --total_timesteps 200000 \
    --learning_rate 3e-4 \
    --n_steps 2048 \
    --batch_size 64 \
    --n_epochs 10 \
    --target_altitude 100.0 \
    --collision_threshold 2.0 \
    --model_name my_model \
    --log_dir ./logs/ \
    --save_dir ./models/
```

### Parametre Açıklamaları

- `--total_timesteps`: Toplam eğitim adımı (ne kadar uzun, o kadar iyi)
- `--learning_rate`: Öğrenme hızı (genelde 1e-4 ile 1e-3 arası)
- `--n_steps`: Her güncellemede toplanan adım sayısı
- `--batch_size`: Batch boyutu
- `--n_epochs`: Her güncellemede kaç epoch
- `--target_altitude`: Hedef irtifa (metre)
- `--collision_threshold`: Çarpışma eşiği (metre)

### TensorBoard ile İzleme

```bash
tensorboard --logdir ./logs/
```

Tarayıcıda `http://localhost:6006` adresini açın.

---

## 🧪 Test Senaryoları

### Tek Bir Modeli Test Et

```bash
python test_scenarios.py \
    --model_path ./models/ppo_flight_best/best_model.zip \
    --n_episodes 10
```

### Özel Senaryolar

1. Senaryo dosyası oluştur (`my_scenarios.json`):

```json
{
  "normal_flight": {
    "collision_threshold": 2.0,
    "target_altitude": 100.0,
    "altitude_tolerance": 20.0,
    "max_roll": 30.0,
    "max_pitch": 30.0,
    "max_episode_steps": 1000,
    "initial_altitude_range": [50.0, 150.0],
    "dt": 0.1
  },
  "extreme_low": {
    "collision_threshold": 2.0,
    "target_altitude": 100.0,
    "altitude_tolerance": 20.0,
    "max_roll": 30.0,
    "max_pitch": 30.0,
    "max_episode_steps": 1000,
    "initial_altitude_range": [5.0, 15.0],
    "dt": 0.1
  }
}
```

2. Test et:

```bash
python test_scenarios.py \
    --model_path ./models/my_model_best/best_model.zip \
    --scenarios my_scenarios.json \
    --n_episodes 20
```

### Varsayılan Senaryolar

Script otomatik olarak şu senaryoları test eder:

1. **normal_flight**: Normal uçuş koşulları
2. **low_altitude_start**: Düşük irtifadan başlama
3. **high_altitude_start**: Yüksek irtifadan başlama
4. **strict_target**: Daha sıkı hedef toleransı
5. **long_episode**: Uzun episode (2000 step)

### Çıktılar

- `test_results_YYYYMMDD_HHMMSS.csv`: Detaylı sonuçlar
- `test_results_YYYYMMDD_HHMMSS.png`: Görselleştirme

---

## 🔧 Hyperparameter Tuning

### Otomatik Grid Search

```bash
python hyperparameter_tuning.py \
    --total_timesteps 50000 \
    --test_episodes 5 \
    --output_dir ./tuning_results/
```

Bu komut varsayılan parametre grid'ini test eder:
- `learning_rate`: [1e-4, 3e-4, 1e-3]
- `n_steps`: [1024, 2048]
- `batch_size`: [32, 64]
- `n_epochs`: [5, 10]

**Toplam 3 × 2 × 2 × 2 = 24 kombinasyon test edilir!**

### Özel Parametre Grid'i

1. Config dosyası oluştur (`tuning_config.json`):

```json
{
  "param_grid": {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "n_steps": [1024, 2048, 4096],
    "batch_size": [32, 64, 128],
    "n_epochs": [5, 10, 20]
  },
  "base_config": {
    "target_altitude": 100.0,
    "collision_threshold": 2.0
  }
}
```

2. Çalıştır:

```bash
python hyperparameter_tuning.py \
    --config_file tuning_config.json \
    --total_timesteps 50000 \
    --test_episodes 5
```

### Çıktılar

- `tuning_results_YYYYMMDD_HHMMSS.csv`: Tüm kombinasyonların sonuçları
- `best_config_YYYYMMDD_HHMMSS.json`: En iyi konfigürasyon
- `./tuning_results/models/`: Tüm eğitilmiş modeller

### En İyi Modeli Bulma

CSV dosyasını aç ve `mean_reward` sütununa göre sırala:

```python
import pandas as pd
df = pd.read_csv('tuning_results_YYYYMMDD_HHMMSS.csv')
df_sorted = df.sort_values('mean_reward', ascending=False)
print(df_sorted.head(10))
```

---

## 📊 Model Karşılaştırma

### İki Modeli Karşılaştır

```bash
python compare_models.py \
    --models ./models/model1_best/best_model.zip \
            ./models/model2_best/best_model.zip \
    --names "Model 1" "Model 2" \
    --n_episodes 10
```

### Birden Fazla Modeli Karşılaştır

```bash
python compare_models.py \
    --models \
        ./models/model1_best/best_model.zip \
        ./models/model2_best/best_model.zip \
        ./models/model3_best/best_model.zip \
    --names "LR=1e-4" "LR=3e-4" "LR=1e-3" \
    --n_episodes 20
```

### Çıktılar

- `model_comparison.csv`: Detaylı karşılaştırma
- `model_comparison.png`: Görselleştirme

---

## 🎯 Önerilen Workflow

### 1. İlk Deneme

```bash
# Hızlı bir test eğitimi
python train_ppo.py --total_timesteps 50000 --model_name test_run

# Test et
python test_scenarios.py \
    --model_path ./models/test_run_best/best_model.zip \
    --n_episodes 5
```

### 2. Hyperparameter Tuning

```bash
# Grid search (uzun sürebilir!)
python hyperparameter_tuning.py \
    --total_timesteps 50000 \
    --test_episodes 5
```

### 3. En İyi Parametrelerle Eğit

```bash
# Tuning sonuçlarından en iyi parametreleri al
python train_ppo.py \
    --total_timesteps 500000 \
    --learning_rate 3e-4 \
    --n_steps 2048 \
    --batch_size 64 \
    --n_epochs 10 \
    --model_name final_model
```

### 4. Kapsamlı Test

```bash
python test_scenarios.py \
    --model_path ./models/final_model_best/best_model.zip \
    --n_episodes 50
```

### 5. Farklı Modelleri Karşılaştır

```bash
python compare_models.py \
    --models \
        ./models/model_v1_best/best_model.zip \
        ./models/model_v2_best/best_model.zip \
        ./models/final_model_best/best_model.zip \
    --names "V1" "V2" "Final" \
    --n_episodes 20
```

---

## 📈 Metrikler ve Değerlendirme

### Önemli Metrikler

1. **Mean Reward**: Ortalama ödül (yüksek = iyi)
2. **Success Rate**: Başarı oranı (collision olmadan tamamlanan episode'lar)
3. **Collision Rate**: Çarpışma oranı (düşük = iyi)
4. **Mean Episode Length**: Ortalama episode uzunluğu

### İyi Bir Model

- ✅ Yüksek mean reward (>0)
- ✅ Yüksek success rate (>80%)
- ✅ Düşük collision rate (<10%)
- ✅ Tüm senaryolarda tutarlı performans

---

## 🐛 Sorun Giderme

### Model Bulunamıyor

```bash
# Model dosyasının yolunu kontrol et
ls -la ./models/*/best_model.zip
```

### CUDA/GPU Hatası

CPU'da çalıştırmak için:

```python
# train_ppo.py içinde
model = PPO('MlpPolicy', env, device='cpu', **ppo_config)
```

### Bellek Hatası

- `batch_size`'ı küçült (32 veya 16)
- `n_steps`'i küçült (1024)
- `total_timesteps`'i azalt

---

## 💡 İpuçları

1. **Küçük başla**: İlk denemelerde `total_timesteps=50000` yeterli
2. **TensorBoard kullan**: Eğitim ilerlemesini izle
3. **Checkpoint'leri kontrol et**: Her 50000 step'te model kaydedilir
4. **Farklı senaryolar test et**: Modelin genelleme yeteneğini ölç
5. **Hyperparameter tuning yap**: En iyi kombinasyonu bul

---

## 📚 Sonraki Adımlar

1. ✅ Model eğitimi tamamlandı
2. ✅ Test senaryoları çalıştırıldı
3. ✅ En iyi model seçildi
4. ⏭️ Gerçek uygulamada kullanıma hazır!

---

**Not**: Bu workflow, statik FlightGear verisiyle değil, **Gym environment ile interaktif eğitim** yapar. Bu, RL için doğru yaklaşımdır! 🚀

