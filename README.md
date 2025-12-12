📌 RL-Based Autonomous Maneuver and Obstacle Avoidance System for Helicopters

Bu proje; taktik, askeri ve sivil helikopterler için RL (Reinforcement Learning) tabanlı, tamamen otonom bir engel kaçınma ve manevra sistemi geliştirmeyi amaçlar. Sistem; sensör verilerine dayanarak çevreyi algılar, PPO algoritmasıyla optimal kontrol komutları üretir ve çeşitli zorluk seviyelerinde güvenli uçuş kabiliyeti sağlar.

🚁 Projenin Amacı

Bu projenin ana hedefi:
Helikopterin çevre koşullarını sürekli analiz edebilmesi
Engelleri algılayıp güvenli manevralar oluşturması
Farklı rüzgar, basınç ve sensör gürültüsü durumlarında stabil uçabilmesi
RL tabanlı otonom kontrol sistemi üretmek
AirSim/FlightGear gibi gerçekçi bir simülasyon ortamında otonom uçuş gerçekleştirmek

Bu sistem;
✔ Otonom keşif–gözetleme görevleri
✔ Riskli uçuş bölgelerinde insansız görev yürütme
✔ Arama–kurtarma operasyonları
✔ Eğitim ve test simülasyonları
için kullanılabilir.

🧠 Kullanılan Yöntemler ve Teknolojiler
Bu çalışmada RL yaklaşımıyla PPO (Proximal Policy Optimization) kullanılmıştır.
Teknik kavramlar:
Reinforcement Learning + PPO
Domain Randomization
Curriculum Learning
Continuous Action Space Control
Sensor Noise Modeling
Wind / Turbulence Simulation
Reward Shaping

🛰️ Sistem Mimarisi
Proje 4 ana katmandan oluşur:

1) Perception Layer
IMU sensörü (gürültülü veri)
Pozisyon, hız, yönelim verisi
Rüzgar ve çevresel modeller

2) Simulation Layer
AirSim / FlightGear helikopter modeli
Çevresel faktörlerin rastgeleleştirilmesi
Obstacle ve terrain senaryoları

4) RL Environment (Gym Wrapper)
Gözlem vektörü oluşturma
Aksiyon dönüşümları
Reward fonksiyonu
Episode başlangıç/bitiş şartları
State normalization

5) Control Layer
PPO agent
Actor–Critic mimarisi
Checkpoint, evaluation, loglama
