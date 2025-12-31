# 🔍 Helikopter Davranış Analizi Kılavuzu

Bu kılavuz, eğitilmiş ajanın gerçekten bir helikopter gibi davranıp davranmadığını analiz etmek için hazırlanmıştır.

## 📊 Analiz Yöntemleri

### 1. İstatistiksel Karşılaştırma

Ajanın davranışını gerçek helikopter log verileriyle karşılaştırır:

```bash
python analyze_helicopter_behavior.py \
    --model_path ./models_3d/full_3d_model_best/best_model.zip \
    --n_episodes 20
```

Bu analiz şunları karşılaştırır:
- **Altitude AGL**: Yükseklik dağılımı
- **Roll Angle**: Yatış açısı dağılımı
- **Pitch Angle**: Yükseliş/dalış açısı dağılımı
- **Altitude Rate**: İrtifa değişim hızı
- **Overall Similarity Score**: Genel benzerlik skoru (0-100%)

### 2. Uçuş Dinamikleri Analizi

Ajanın uçuş karakteristiklerini analiz eder:
- Hız profili
- Attitude stabilitesi
- Altitude kontrolü
- Maneuver kalitesi

### 3. Görselleştirme

Ajan ve gerçek helikopter verilerini yan yana gösterir:
- Dağılım karşılaştırmaları
- Trajectory karşılaştırmaları
- Zaman serisi analizleri

## 📈 Çıktılar

### 1. Similarity Score

```
OVERALL SIMILARITY SCORE: 75.3%
```

**Yorumlama:**
- **>70%**: ✅ Ajan gerçekçi davranıyor
- **50-70%**: ⚠️ Orta seviye, iyileştirme gerekli
- **<50%**: ❌ Gerçekçi değil, önemli iyileştirme gerekli

### 2. Detaylı Rapor

`behavior_analysis/behavior_report.txt` dosyası:
- İstatistiksel karşılaştırmalar
- Uçuş dinamikleri analizi
- Öneriler ve iyileştirme tavsiyeleri

### 3. Görselleştirmeler

`behavior_analysis/behavior_comparison.png`:
- 6 farklı karşılaştırma grafiği
- Ajan vs gerçek helikopter

## 🔧 Log Verilerini Daha İyi Kullanma

### Mevcut Kullanım

Şu anda log verileri:
- ✅ Başlangıç durumları için kullanılıyor (attitude, velocity)
- ✅ Gerçekçi helikopter davranışı sağlıyor

### Geliştirme Önerileri

#### 1. Daha Fazla Log Verisi

```python
# Birden fazla log dosyası kullan
log_files = ["fg_log1.csv", "fg_log2.csv", "fg_log3.csv"]
# Daha fazla çeşitlilik = daha iyi öğrenme
```

**Faydaları:**
- Daha fazla uçuş senaryosu
- Daha çeşitli başlangıç durumları
- Daha iyi genelleme

#### 2. Weighted Sampling

```python
# İlginç durumları daha sık seç
# Örnek: Yüksek roll/pitch durumları
interesting_states = log_states[abs(log_states['roll']) > 15]
```

**Faydaları:**
- Zor senaryolara daha fazla maruz kalma
- Daha iyi manevra öğrenme

#### 3. Curriculum Learning

```python
# Kolaydan zora
# 1. Level flight (log verilerinden)
# 2. Gentle turns
# 3. Complex maneuvers
```

#### 4. Reward Shaping

```python
# Log verilerine benzer davranışları ödüllendir
if abs(roll) < log_stats['roll']['mean'] + log_stats['roll']['std']:
    reward += 0.1  # Realistic behavior bonus
```

## 📊 Log Verisi Kalitesi Analizi

Log verilerinizin kalitesini analiz edin:

```bash
python improve_with_log_data.py
```

Bu analiz:
- Veri hacmini kontrol eder
- State çeşitliliğini ölçer
- Uçuş fazlarını analiz eder
- İyileştirme önerileri sunar

## 🎯 İyileştirme Stratejisi

### Adım 1: Mevcut Davranışı Analiz Et

```bash
python analyze_helicopter_behavior.py \
    --model_path ./models_3d/full_3d_model_best/best_model.zip
```

### Adım 2: Log Verisi Kalitesini Kontrol Et

```bash
python improve_with_log_data.py
```

### Adım 3: İyileştirmeleri Uygula

1. **Daha fazla log verisi topla** (farklı senaryolar)
2. **Weighted sampling ekle** (ilginç durumları önceliklendir)
3. **Reward function'ı fine-tune et** (log istatistiklerine göre)
4. **Daha uzun eğitim** (log verileriyle daha fazla deneyim)

### Adım 4: Tekrar Analiz Et

İyileştirmelerden sonra tekrar analiz edin ve karşılaştırın.

## 📈 Log Verisi Detayı ve Gelişim

### Soru: Daha Detaylı Log Verisi Gelişimi Artırır mı?

**Evet!** Daha detaylı log verisi şunları sağlar:

1. **Daha Çeşitli Senaryolar**
   - Farklı hava koşulları
   - Farklı manevralar
   - Farklı yükler

2. **Daha İyi Başlangıç Durumları**
   - Daha gerçekçi initial states
   - Daha fazla çeşitlilik

3. **Daha İyi Genelleme**
   - Daha fazla veri = daha iyi öğrenme
   - Daha az overfitting

### Önerilen Log Verisi İçeriği

**Minimum:**
- Altitude, Roll, Pitch, Heading
- Altitude rate
- Time stamps

**İdeal (Daha Detaylı):**
- Velocity (vx, vy, vz)
- Angular rates (roll_rate, pitch_rate, yaw_rate)
- Control inputs (throttle, cyclic, collective)
- Engine parameters
- Wind conditions

### Log Verisi Kullanım Örnekleri

#### Örnek 1: Basit Kullanım (Mevcut)
```python
# Sadece başlangıç durumları
use_log_data=True
```

#### Örnek 2: Gelişmiş Kullanım
```python
# Weighted sampling + curriculum learning
use_log_data=True
weighted_sampling=True
curriculum_learning=True
```

#### Örnek 3: En İyi Kullanım
```python
# Behavioral cloning + RL fine-tuning
# 1. Pre-train with log data (imitation learning)
# 2. Fine-tune with RL
```

## 🔬 Metrikler ve Değerlendirme

### Önemli Metrikler

1. **Similarity Score**: Genel benzerlik (hedef: >70%)
2. **Attitude Stability**: Roll/pitch stabilitesi (hedef: <10°)
3. **Altitude Control**: Altitude variance (hedef: <50m)
4. **Speed Profile**: Hız dağılımı (log verilerine benzer)

### Başarı Kriterleri

✅ **İyi Ajan:**
- Similarity Score > 70%
- Attitude stability < 10°
- Smooth trajectories
- Realistic maneuvers

⚠️ **Orta Ajan:**
- Similarity Score 50-70%
- Attitude stability 10-20°
- Some unrealistic behaviors

❌ **Kötü Ajan:**
- Similarity Score < 50%
- Attitude stability > 20°
- Erratic behavior

## 💡 Pratik İpuçları

1. **Düzenli Analiz**: Her eğitim sonrası analiz yapın
2. **Karşılaştırma**: Farklı modelleri karşılaştırın
3. **İteratif İyileştirme**: Analiz → İyileştirme → Tekrar Eğitim
4. **Log Verisi Çeşitliliği**: Farklı senaryolardan log verisi toplayın

## 📚 Sonraki Adımlar

1. ✅ Mevcut modeli analiz et
2. ✅ Log verisi kalitesini kontrol et
3. ⏭️ İyileştirmeleri uygula
4. ⏭️ Tekrar eğit ve analiz et
5. ⏭️ En iyi modeli seç

---

**Önemli**: Log verileri sadece başlangıç durumları için değil, **gerçekçi helikopter davranışı öğrenmek** için kullanılıyor! 🚁

