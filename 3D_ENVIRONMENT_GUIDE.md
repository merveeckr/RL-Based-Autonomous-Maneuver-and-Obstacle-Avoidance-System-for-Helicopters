# 🚁 3D Flight Environment Kullanım Kılavuzu

Bu kılavuz, 3D ortamda helikopter uçurma ve engellerden kaçınma için hazırlanmıştır.

## 📋 Özellikler

### ✅ Log Verileri Entegrasyonu
- **FlightGear log verileri (`fg_log2.csv`) kullanılıyor!**
- Her episode, gerçek uçuş verilerinden başlangıç durumlarıyla başlar
- Log verilerinden alınan bilgiler:
  - **Pozisyon**: Altitude (z), Longitude (x yaklaşımı)
  - **Attitude**: Roll, Pitch, Heading (Yaw)
  - **Hız**: Altitude rate (vz), heading'den tahmin edilen vx, vy

### 🎯 3D Environment Özellikleri
- **3D Pozisyon**: [x, y, z] koordinatları
- **3D Hız**: [vx, vy, vz] vektörü
- **Attitude**: Roll, Pitch, Yaw açıları
- **Engeller**: 3D silindirik engeller
- **Hedef**: 3D uzayda hedef nokta
- **Görselleştirme**: Matplotlib 3D rendering

## 🚀 Hızlı Başlangıç

### 1. Environment'ı Test Et

```bash
python flight_env_3d.py
```

Bu komut:
- 3D environment'ı oluşturur
- 5 engel yerleştirir
- Log verilerinden başlangıç durumu seçer
- 50 random step çalıştırır
- 3D görselleştirme gösterir

### 2. Model Eğit

```bash
python train_3d_ppo.py --total_timesteps 200000 --num_obstacles 5
```

Bu komut:
- 3D environment'da PPO modeli eğitir
- Log verilerini kullanarak gerçekçi başlangıç durumları oluşturur
- Modeli `./models_3d/` klasörüne kaydeder

### 3. Eğitilmiş Modeli Test Et

```bash
python test_3d_environment.py \
    --model_path ./models_3d/ppo_3d_best/best_model.zip \
    --num_episodes 5
```

## 📊 State Space (11D)

```python
state = [
    x, y, z,              # 3D pozisyon (metre)
    vx, vy, vz,           # 3D hız (m/s)
    roll, pitch, yaw,     # Attitude (derece)
    altitude_rate         # İrtifa değişim hızı (m/s)
]
```

## 🎮 Action Space (4D)

```python
action = [
    roll_command,    # [-1, 1] Roll kontrolü
    pitch_command,   # [-1, 1] Pitch kontrolü
    yaw_command,     # [-1, 1] Yaw kontrolü
    throttle_command # [-1, 1] Throttle kontrolü
]
```

## 🎁 Reward Function

- **Progress Reward**: Hedefe yaklaşma ödülü
- **Goal Reward**: Hedefe ulaşma ödülü (+100)
- **Obstacle Penalty**: Engellere yaklaşma cezası
- **Collision Penalty**: Çarpışma cezası (-100)
- **Stability Reward**: Düşük attitude açıları için küçük ödül

## 🔧 Parametreler

### Environment Parametreleri

```python
env = FlightControlEnv3D(
    world_size=(500.0, 500.0, 200.0),  # [x, y, z] boyutları (metre)
    num_obstacles=5,                    # Engel sayısı
    obstacle_radius_range=(10.0, 30.0), # Engel yarıçap aralığı
    obstacle_height_range=(50.0, 150.0), # Engel yükseklik aralığı
    max_speed=50.0,                     # Maksimum hız (m/s)
    max_episode_steps=2000,            # Maksimum episode uzunluğu
    use_log_data=True,                  # Log verilerini kullan
    log_data_path='fg_log2.csv'        # Log dosyası yolu
)
```

## 📈 Eğitim Örnekleri

### Temel Eğitim

```bash
python train_3d_ppo.py \
    --total_timesteps 200000 \
    --num_obstacles 5 \
    --model_name my_3d_model
```

### Zor Senaryo (Daha Fazla Engel)

```bash
python train_3d_ppo.py \
    --total_timesteps 500000 \
    --num_obstacles 10 \
    --model_name hard_3d_model
```

### Hızlı Test Eğitimi

```bash
python train_3d_ppo.py \
    --total_timesteps 50000 \
    --num_obstacles 3 \
    --model_name quick_test_3d
```

## 🧪 Test Senaryoları

### Görselleştirmeli Test

```bash
python test_3d_environment.py \
    --model_path ./models_3d/ppo_3d_best/best_model.zip \
    --num_episodes 5
```

### Görselleştirme Olmadan Test

```bash
python test_3d_environment.py \
    --model_path ./models_3d/ppo_3d_best/best_model.zip \
    --num_episodes 10 \
    --no_render
```

### Zor Senaryo Testi

```bash
python test_3d_environment.py \
    --model_path ./models_3d/ppo_3d_best/best_model.zip \
    --num_episodes 5 \
    --num_obstacles 10
```

## 📊 Log Verileri Kullanımı

### Nasıl Çalışıyor?

1. **Log Verileri Yüklenir**: `fg_log2.csv` dosyasından 876 state yüklenir
2. **Her Episode Başlangıcı**: Log verilerinden rastgele bir state seçilir
3. **Başlangıç Durumu**:
   - **Z pozisyonu**: Log'dan `altitude_agl`
   - **X pozisyonu**: Log'dan `longitude` (normalize edilmiş)
   - **Y pozisyonu**: Rastgele (latitude yok)
   - **Attitude**: Log'dan `roll`, `pitch`, `heading`
   - **Hız**: Log'dan `altitude_rate` ve heading'den tahmin edilen vx, vy

### Log Verilerinden Alınan Bilgiler

| Log Verisi | 3D Environment Kullanımı |
|------------|-------------------------|
| `Altitude - GroundAltitude` | Z pozisyonu (altitude_agl) |
| `Longitude` | X pozisyonu (yaklaşık) |
| `Roll` | Roll açısı |
| `Pitch` | Pitch açısı |
| `Heading` | Yaw açısı |
| `altitude_rate` | Vz hızı |

## 🎯 Başarı Metrikleri

- **Goal Reached**: Hedefe 10m içinde ulaşma
- **Success Rate**: Hedefe ulaşan episode oranı
- **Collision Rate**: Çarpışma oranı
- **Mean Reward**: Ortalama ödül
- **Episode Length**: Episode uzunluğu

## 💡 İpuçları

1. **İlk Eğitim**: Az engelle başla (3-5 engel)
2. **Log Verileri**: Her zaman aktif tut (daha gerçekçi)
3. **Görselleştirme**: Eğitim sırasında kapat (yavaşlatır)
4. **Test**: Görselleştirmeyle test et (modelin davranışını gör)
5. **TensorBoard**: Eğitim ilerlemesini izle

## 🐛 Sorun Giderme

### Log Verileri Yüklenmiyor

```bash
# Dosyanın varlığını kontrol et
ls -la fg_log2.csv

# Environment'ı test et
python -c "from flight_env_3d import FlightControlEnv3D; env = FlightControlEnv3D(use_log_data=True); obs, info = env.reset(); print('OK')"
```

### Görselleştirme Çalışmıyor

- Matplotlib backend sorunları için:
```python
import matplotlib
matplotlib.use('TkAgg')  # veya 'Qt5Agg'
```

### Model Bulunamıyor

```bash
# Model dosyasını kontrol et
ls -la ./models_3d/*/best_model.zip
```

## 📚 Sonraki Adımlar

1. ✅ 3D environment hazır
2. ✅ Log verileri entegre edildi
3. ⏭️ Model eğitimi başlat
4. ⏭️ Test senaryoları çalıştır
5. ⏭️ En iyi modeli seç

---

**Önemli**: Log verileri (`fg_log2.csv`) her zaman kullanılıyor! Bu sayede model gerçek uçuş durumlarından öğreniyor. 🚁

