# Deprem Şiddeti Tahmin Projesi (PyTorch)

Bu proje, ivmeölçer verilerinden deprem şiddetini tahmin eden bir PyTorch modeli eğitmek ve ayrıntılı hata analizleri üretmek için hazırlanmıştır.

## Hedef
- 2022'den beri biriktirdiğiniz deprem verileriyle model eğitimi
- GPU (AMD RX 7700 XT / ROCm) ile hızlandırılmış eğitim
- Test setinde:
  - Tahmin edilen şiddet
  - Toplam işleme süresi
  - Doğru/yanlış sınıflandırma
  - Yanlış alarm (false alarm)
  - Kaçırılan deprem (missed detection)
  - Tüm metriklerin yüzdeye çevrilmiş raporu

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ROCm kurulu bir sistemde PyTorch'un ROCm build'i gerekir. `torch.cuda.is_available()` true olmalıdır.

## Veri Formatı
CSV beklenir. Örnek sütunlar:

- `timestamp` (opsiyonel, ama varsa zamansal ayırma için iyi olur)
- Özellik sütunları (ivmeölçer özetleri):
  - `acc_x_rms`, `acc_y_rms`, `acc_z_rms`, `pga`, `pgv`, `dominant_freq`, ...
- Hedef sütun:
  - `intensity` (ör. sürekli şiddet değeri, magnitüd benzeri)

## Eğitim + Değerlendirme

```bash
python train_and_evaluate.py \
  --data-path data/earthquakes.csv \
  --target-col intensity \
  --time-col timestamp \
  --event-threshold 3.5 \
  --epochs 80 \
  --batch-size 256 \
  --learning-rate 1e-3
```

### Parametreler
- `--feature-cols`: Özellik sütunlarını elle vermek isterseniz (boşsa otomatik seçer)
- `--val-ratio`: doğrulama oranı
- `--test-ratio`: test oranı
- `--hidden-dims`: MLP katmanları (örn: `128 64 32`)
- `--model-out`: model kayıt dosyası
- `--metrics-out`: JSON rapor dosyası

## Çıktılar
- `artifacts/model.pt`: Eğitilmiş model
- `artifacts/metrics.json`: Tüm metrikler ve süre raporu

## Üretilen Metrikler

Regresyon:
- MAE
- RMSE
- R²

Olay algılama (eşik bazlı):
- Accuracy
- Precision
- Recall (Detection Rate)
- F1
- True Positive / True Negative / False Positive / False Negative
- False Alarm Rate (%)
- Miss Rate (%)
- Doğru tahmin oranı (%)

Performans:
- Test örneği sayısı
- Toplam işleme süresi (s)
- Saniye başına işlenen örnek (samples/sec)

## Notlar
- AMD GPU'da eğitim için ROCm kurulumu doğrulanmalı.
- İlk etapta bu model bir MLP tabanlı baseline'dır; daha sonra zaman serisi için 1D-CNN/LSTM/Transformer'a geçebilirsiniz.
