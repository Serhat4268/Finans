# Finansal Varlık Yön Tahmini ve Gerçek Zamanlı Sinyal Üretim Platformu

Akademik Makine Öğrenmesi & Finansal Veri Bilimi Projesi (4 Haftalık Geliştirme Yol Haritası)

## 📁 Proje Mimarisi

```text
Fİnans/
├── requirements.txt         # Proje bağımlılıkları (pandas, yfinance, ta, scikit-learn, xgboost, streamlit vb.)
├── run_phase1.py            # Faz 1 demo ve boru hattı doğrulama scripti
├── run_phase2.py            # Faz 2 model eğitimi, TimeSeriesSplit CV ve sinyal simülasyonu
├── run_phase3.py            # Faz 3 SQLite sinyal loglama ve backtest yürütme scripti
├── src/
│   ├── __init__.py
│   ├── config.py            # Model, indikatör ve veri parametreleri konfigürasyonu
│   ├── data_loader.py       # yfinance veri çekme, OHLCV tutarlılık kontrolü ve Parquet önbellekleme
│   ├── feature_engine.py    # ta indikatörleri, Supertrend, volatilite ve sahte kırılım (fakeout) filtreleri
│   ├── model_trainer.py     # TimeSeriesSplit, XGBoost & LightGBM eğitimi ve serileştirme (.pkl)
│   ├── signal_manager.py    # SQLite veritabanı yönetimi ve dinamik ATR risk hesaplamaları
│   └── backtester.py        # Geriye dönük test motoru (Win Rate, Profit Factor, MDD, Sharpe)
├── tests/
│   ├── test_pipeline.py     # Faz 1 birim testleri
│   ├── test_model.py        # Faz 2 birim testleri
│   └── test_database.py     # Faz 3 birim testleri
├── models/                  # Serileştirilmiş yapay zeka modelleri (.pkl)
├── database/                # SQLite sinyal ve performans veritabanı (signals.db)
└── data/
    └── raw/                 # İndirilen ham ve temizlenmiş Parquet önbellek dosyaları
```

## 🚀 Hızlı Başlangıç

### ⚡ Tek Komutla Çalıştırma (Önerilen)
Artık 5 farklı komutu tek tek yazmanıza gerek yoktur:

**Windows Çift Tıklama:**
Klasördeki `start.bat` dosyasına çift tıklayarak veya terminalden:
```powershell
.\start.bat
```

**Python ile:**
```powershell
# Doğrudan Karar Destek Arayüzünü (Streamlit) açar (eksik model varsa otomatik eğitilir)
python run.py

# VEYA 5 adımlık tüm akademik boru hattını (Testler -> Faz 1 -> Faz 2 -> Faz 3 -> Streamlit) sırayla çalıştırır:
python run.py --all

# Sadece eğitim boru hattını (Faz 1-3) arayüz açmadan çalıştırır:
python run.py --train

# Sadece birim testleri çalıştırır:
python run.py --test
```

---

### 🔬 Adım Adım Geliştirici Kılavuzu (Manuel)

### 1. Tüm Birim Testleri Çalıştırma (Faz 1, Faz 2 & Faz 3)
```powershell
python -m unittest discover tests
```

### 2. Veri Hattını ve İndikatör Motorunu Test Etme (Faz 1)
```powershell
python run_phase1.py BTC-USD
```

### 3. Model Eğitimi, Time-Series CV ve Sinyal Üretimi (Faz 2)
```powershell
python run_phase2.py BTC-USD
```

### 4. SQLite Sinyal Loglama ve Backtest Simülasyonu (Faz 3)
```powershell
python run_phase3.py BTC-USD
```

Bu adım:
- Dinamik 1.5x ATR Stop-Loss ve 1:2 Risk/Reward kâr al hedeflerini hesaplar.
- Üretilen tüm sinyalleri olasılık dağılımlarıyla `database/signals.db` veritabanına loglar.
- Çubuk bazlı simülasyonla kâr faktörünü, kazanma oranını ve sermaye getirisini ölçer.

### 5. Streamlit İnteraktif Karar Destek Arayüzünü Başlatma (Faz 4)
```powershell
streamlit run app.py
```
Arayüz tarayıcınızda (`http://localhost:8501`) açılır:
- Canlı şamdan grafikleri, Bollinger Bantları, Supertrend ve Hacim göstergeleri.
- % Al, % Nötr, % Sat olasılık ibreleri ve model güven skoru.
- Dinamik Stop-Loss ve Kâr Al seviyesi göstergeleri.
- Sahte Kırılım (Boğa/Ayı tuzağı) uyarı bildirimleri.
- SQLite sinyal arşivi ve sermaye büyüme eğrisi (Equity Curve).

## 📊 Faz 1 Kapsamında Üretilen Öznitelikler

- **Trend:** EMA (20, 50, 200), Fiyat-EMA normalize mesafeleri, EMA Trend Kesişimi, MACD, MACD Signal, MACD Histogram İvmesi.
- **Momentum:** RSI (14), RSI Değişimi ($\Delta \text{RSI}$), Aşırı Alım/Satım bayrakları, Stokastik Osilatör (%K, %D).
- **Volatilite:** ATR (14), Normalize ATR (NATR: Yüzde cinsinden volatilite), Bollinger Bantları (Üst, Alt, Orta, Bant Genişliği, %B), Bollinger Sıkışması (Squeeze).
- **Supertrend:** ATR ve bant çarpanı tabanlı dinamik trend yönü (`1`: Alış/Boğa, `-1`: Satış/Ayı).
- **Hacim Teyidi:** OBV (On-Balance Volume), 20 periyotluk Hacim Ortalaması, Hacim Patlama Katsayısı (`Volume_Ratio`).
- **Sahte Kırılım (Fakeout) Filtreleri:**
  - `Upper_Wick_Ratio`: Mum üst fitil baskısı
  - `Lower_Wick_Ratio`: Mum alt fitil toparlanması
  - `Fakeout_High_Risk`: Hacimsiz kırılım veya tepe reddi (Boğa Tuzağı)
  - `Fakeout_Low_Risk`: Hacimsiz çöküş veya dip toparlanması (Ayı Tuzağı)
- **Faz 2 Hazırlığı:** $N$ bar sonraki getiriye göre 3 sınıflı hedef etiket (`1`: Al, `0`: Nötr, `-1`: Sat).

---

## 🤖 Yapay Zeka Modelleri: XGBoost & LightGBM

Bu projede fiyat yön tahmini için iki farklı **Gradient Boosting (Gradyan Artırma)** algoritması kullanılmaktadır. Her ikisi de klasik "eğer RSI > 70 ise sat" gibi sabit kurallara değil; onlarca teknik indikatörün **birleşik istatistiksel örüntüsünü** öğrenerek karar üretir.

### Gradient Boosting Nedir?
Her iki model de aynı temel fikre dayanır: Çok sayıda **zayıf öğrenici (karar ağacı)** sırayla birbirinin hatalarını düzelterek birleşir ve güçlü bir tahmin motoru oluşturur. Tek büyük bir karar ağacının aksine binlerce küçük ağaç birbirini tamamlar.

```
[Öznitelikler] → Ağaç 1 → Ağaç 2 → ... → Ağaç N → [AL / NÖTR / SAT]
                   hata      hatayı          hatayı
                   üretir    düzeltir        düzeltir
```

---

### ⚡ XGBoost (Extreme Gradient Boosting)

| Özellik | Değer |
| :--- | :--- |
| **Geliştirici** | T. Chen – DMLC / Washington Üniversitesi (2016) |
| **Hız** | CPU üzerinde paralel ağaç inşası; oldukça hızlı |
| **Bu projede parametreler** | 120 ağaç, max derinlik 4, öğrenme hızı 0.04 |
| **Kayıp fonksiyonu** | `mlogloss` (3 sınıf: AL / NÖTR / SAT) |
| **Düzenlileştirme** | L1 (Lasso) + L2 (Ridge) → aşırı öğrenmeyi engeller |

**Bu projede ne yapar?**
- EMA, RSI, Bollinger, Supertrend, ATR, OBV gibi **30+ teknik özniteliği** bir arada analiz eder.
- Her bar için **"N gün sonra fiyat yukarı mı, aşağı mı, yatay mı gider?"** sorusuna olasılıkla yanıt verir.
- `src/model_trainer.py` içinde `XGBClassifier` olarak kullanılır; eğitilmiş hali `models/model_*_xgboost.pkl` dosyasına kaydedilir.

**Ne zaman güçlüdür?**
- Öznitelikler arasında **doğrusal olmayan karmaşık ilişkiler** olduğunda.
- Eksik veri içeren öznitelik setlerinde bile kararlı çalışır.

---

### 🌿 LightGBM (Light Gradient Boosting Machine)

| Özellik | Değer |
| :--- | :--- |
| **Geliştirici** | Microsoft Research (2017) |
| **Hız** | XGBoost'a kıyasla ~10x daha hızlı eğitim; bellek dostu |
| **Bu projede parametreler** | `LGBMClassifier` varsayılan + TimeSeriesSplit ile optimize |
| **Ağaç büyüme stratejisi** | **Leaf-wise** (yaprak bazlı) → daha derin, daha isabetli ağaçlar |
| **Düzenlileştirme** | `min_child_samples`, `num_leaves` ile aşırı öğrenme kontrolü |

**Bu projede ne yapar?**
- XGBoost ile **aynı öznitelik setini** farklı bir iç algoritmayla öğrenir.
- Özellikle **büyük veri setlerinde ve çok sayıda öznitelikte** XGBoost'tan daha hızlı ve çoğunlukla daha isabetli sonuç üretir.
- `src/model_trainer.py` içinde `LGBMClassifier` olarak kullanılır; eğitilmiş hali `models/model_*_lightgbm.pkl` dosyasına kaydedilir.
- Arayüzde **varsayılan model olarak** seçilir (çünkü hafıza ve hız açısından daha verimlidir).

**Ne zaman güçlüdür?**
- Zaman serisi gibi **sıralı ve yüksek boyutlu** verilerde.
- Kategori özniteliklerini doğal olarak işleyebilir.

---

### 🔁 İki Modelin Birlikte Kullanılma Sebebi

| | XGBoost | LightGBM |
| :--- | :---: | :---: |
| Eğitim hızı | ✅ Hızlı | ✅✅ Çok hızlı |
| Büyük veri setleri | ✅ İyi | ✅✅ Çok iyi |
| Küçük/orta veri setleri | ✅✅ Çok iyi | ✅ İyi |
| Aşırı öğrenmeye direnç | ✅ Güçlü | ✅ Güçlü |
| Varsayılan model (bu proje) | — | ✅ |

İki modelin ayrı ayrı eğitilip karşılaştırılması, **hangisinin o varlık için daha isabetli sinyal ürettiğini** ölçmeyi ve karşılaştırmalı backtest yapmayı mümkün kılar.

---

### ⏱️ TimeSeriesSplit ile Doğrulama

Her iki model de standart çapraz doğrulama (`k-fold`) yerine **`TimeSeriesSplit`** ile değerlendirilir. Bunun sebebi finansal verideki **zamansal sıranın** bozulmamasıdır:

```
Fold 1:  [Eğitim: Oca–Nis] → [Test: May]
Fold 2:  [Eğitim: Oca–Tem] → [Test: Ağu]
Fold 3:  [Eğitim: Oca–Eki] → [Test: Kas]
...
```

Gelecekteki verinin geçmişi "sızdırması" (lookahead bias) bu yöntemle engellenir.

---

## 🧠 İndikatör Hesaplama Mantığı ve RSI Öznitelik Mühendisliği

### 1. Zaman Dilimi (Timeframe)
- **Varsayılan Aralık:** **Günlük (1D - Daily)** barlar kullanılmaktadır.
- Veri motoru (`src/data_loader.py` ve `app.py`), Yahoo Finance üzerinden günlük kapanış verilerini çekerek hesaplama yapar.
- Modüler mimari sayesinde istenirse konfigürasyondan tek bir parametreyle (`interval="4h"` veya `"1h"`) 4 saatlik ya da saatlik verilere geçirilebilir.

### 2. RSI Matematiksel Hesabı ve Üretilen Öznitelikler
RSI hesaplamasında 14 periyotluk standart **Wilder Smoothing** yöntemi kullanılır:
$$RSI = 100 - \left( \frac{100}{1 + RS} \right) \quad \text{burada} \quad RS = \frac{\text{14 Gündeki Ortalama Kazanç}}{\text{14 Gündeki Ortalama Kayıp}}$$

Yapay zeka modelinin salt bir rakam yerine piyasa dinamiklerini öğrenebilmesi için 4 alt öznitelik üretilir:

| Öznitelik | Tanım / Formül | Finansal & Makine Öğrenmesi Anlamı |
| :--- | :--- | :--- |
| **`RSI`** | Ham 14 periyotluk RSI (0-100) | Varlığın mevcut momentum gücü seviyesi. |
| **`RSI_Delta`** | $RSI_t - RSI_{t-1}$ | **RSI İvmesi:** Momentumun artış veya azalış hızı. |
| **`RSI_Overbought`** | $RSI \ge 70 \rightarrow 1$, değilse $0$ | **Aşırı Alım Bölgesi:** Fiyatın doyum noktasına ulaşıp ulaşmadığı. |
| **`RSI_Oversold`** | $RSI \le 30 \rightarrow 1$, değilse $0$ | **Aşırı Satım Bölgesi:** Satış baskısının tükenme riski. |

### 3. Yapay Zekanın RSI'ı Yorumlama Mekanizması
Geleneksel teknik analizde yapılan *"RSI 70'i aştı hemen sat, 30'un altına indi hemen al"* ezberi yerine, makine öğrenmesi modellerimiz (LightGBM & XGBoost) RSI'ı diğer indikatörlerle çapraz doğrular:

1. **Trend Devamlılığı Teyidi:**
   - **Supertrend = Yükseliş** ve **EMA 20 > EMA 50** durumunda, RSI'ın 60–70 bandında seyretmesi güçlü bir trendin devamı olarak okunur ve **AL (BUY)** olasılığı artırılır.
2. **Sahte Kırılım (Boğa Tuzağı) Filtreleme:**
   - Fiyat direnci veya Üst Bollinger Bandını zorladığı halde $RSI \ge 70$ iken **`RSI_Delta` negatife dönmüşse** (negatif uyumsuzluk) ve `Volume_Ratio < 1.0` (hacim yetersiz) ise sistem bunu bir **Boğa Tuzağı (Fakeout)** olarak etiketler; AL sinyali baskılanarak **SAT / NÖTR** olasılığı ön plana çıkarılır.
