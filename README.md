# Elektrikli Araç Satın Alımlarını Tahmin (Playground Series S6E9)

Kaggle **Playground Series - Season 6, Episode 9** yarışması için geliştirilen çözüm.

- **Yarışma:** https://www.kaggle.com/competitions/playground-series-s6e9
- **Metrik:** ROC-AUC
- **Hedef değişken:** `Will_Buy_EV` (Yes/No)
- **Veri boyutu:** 668,665 satır (train), aynı büyüklükte test seti

## Sonuç Özeti

| Model / Yaklaşım | OOF AUC | Public LB |
|---|---|---|
| HistGradientBoostingClassifier (sklearn) | 0.9397 | – |
| CatBoost (native categorical) | **0.9415** | **0.94129** |
| HistGBM + CatBoost weighted blend (0.3/0.7) | 0.9414 | 0.94081 |
| Rank-average blend | 0.9412 | – |
| Stacking (Logistic Regression meta-model) | 0.9415 | – |

**Final submission olarak seçilenler:** `submission_final.csv` (CatBoost) ve `submission_blend.csv` (yedek).

## Veri Seti

**Sayısal kolonlar:** `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level` (1-5 ordinal skor)

**Kategorik kolonlar:** `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible` (Yes/No), `Subsidy_Available` (Yes/No), `Range_Anxiety_Level` (Low/Medium/High — ordinal)

**Target dağılımı:** %82.5 No, %17.5 Yes (orta düzeyde dengesiz → StratifiedKFold kullanıldı)

**Eksik veri:** Yok.

## Yaklaşım

1. **EDA:** Kolon tipleri, eksik değerler, target dağılımı ve genel istatistikler incelendi.
2. **Target encoding:** `Will_Buy_EV` metin (Yes/No) olduğu için `{Yes: 1, No: 0}` şeklinde sayısala çevrildi.
3. **Ordinal düzeltme:** `Range_Anxiety_Level` kolonu alfabetik değil, gerçek sırasına göre (`Low=0, Medium=1, High=2`) encode edildi.
4. **Feature engineering:**
   - `Total_Charging_Stations` = ev + iş yakınındaki şarj istasyonu sayısı toplamı
   - `Commute_per_Car` = günlük mesafe / sahip olunan araç sayısı
   - `Income_per_Age` = yıllık gelir / yaş
5. **Modelleme:**
   - **HistGradientBoostingClassifier** (sklearn) — kategorikler `OrdinalEncoder` ile encode edildi. macOS'ta LightGBM/XGBoost `libomp` bağımlılığı (Homebrew eksikliği nedeniyle) çalışmadığı için tercih edildi.
   - **CatBoost** — kategorik kolonlar native olarak (ham string haliyle) verildi, ek encoding gerekmedi.
   - 5-fold `StratifiedKFold` (random_state=42) ile cross-validation yapıldı, her iki model için de aynı fold bölünmesi kullanıldı.
6. **Kombinasyon denemeleri:** Basit ortalama, ağırlıklı ortalama, rank-average blend ve Logistic Regression ile stacking denendi. Hiçbiri CatBoost'un tek başına skorunu geçemedi — bu nedenle final submission olarak CatBoost tercih edildi, blend versiyonu yedek final seçim olarak bırakıldı.

## Ortam Kurulumu ve Karşılaşılan Sorunlar

```bash
pip3 install pandas numpy scikit-learn catboost
```

**Not:** Bu makinede (Homebrew kurulu olmayan macOS) hem `lightgbm` hem `xgboost` paketleri `OSError: ... Library not loaded: @rpath/libomp.dylib` hatası verdi. Bu, her iki kütüphanenin native derlemesinin OpenMP çalışma zamanına ihtiyaç duymasından kaynaklanıyor. Çözüm için iki seçenek var:

- Homebrew kurup `brew install libomp` çalıştırmak, **veya**
- (Bu projede tercih edilen) LightGBM yerine sklearn'ün `HistGradientBoostingClassifier`'ını kullanmak — harici bağımlılık gerektirmiyor.

CatBoost'un macOS wheel'i kendi runtime'ını taşıdığı için bu sorunu yaşamadı.

## Dosyalar

- `ev_prediction.ipynb` — EDA, feature engineering, model eğitimi ve submission üretimi adımlarını içeren notebook.
- `train.csv`, `test.csv`, `sample_submission.csv` — yarışma veri setleri.
- `submission.csv` — ilk baseline (HistGBM).
- `submission_blend.csv` — HistGBM + CatBoost ağırlıklı blend (final seçimlerden biri).
- `submission_final.csv` — CatBoost tahminleri (en iyi tekil model, final seçimlerden biri).

## Nasıl Çalıştırılır

1. `train.csv`, `test.csv`, `sample_submission.csv` dosyalarını notebook ile aynı klasöre koy.
2. Gerekli paketleri kur (`pip3 install pandas numpy scikit-learn catboost`).
3. `ev_prediction.ipynb` içindeki hücreleri sırayla çalıştır.
4. Üretilen `submission_final.csv` dosyasını Kaggle'a yükle.
