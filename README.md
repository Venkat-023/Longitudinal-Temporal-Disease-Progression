# Heart Disease Temporal Analysis — Deep Learning Pipeline

A longitudinal deep-learning pipeline that predicts **cardiac condition progression**
(worsening vs. stable/improving) across ICU admissions, using **Bidirectional LSTM
with attention**, **GRU**, and **Transformer** models.

Supports three data modes:
- 🧪 **Synthetic** — 500 patients, generated locally, no setup required
- 🏥 **MIMIC-III Demo** — 100 ICU patients (open-access, no credentials)
- 🏥 **MIMIC-IV 2.1** — ~300,000 ICU patients (full dataset, 67 GB, via Kaggle)

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Synthetic data pipeline (no setup required)
```bash
python src/run_local_pipeline.py --num_patients 500 --num_timepoints 6
python src/utils/visualize_data.py
python src/models/train_lstm.py
python src/models/train_gru.py
python src/models/train_transformer.py
python src/utils/visualize_results.py
```

### 3. Real clinical data pipeline — MIMIC-III Demo (open-access)
```bash
python src/run_real_pipeline.py
```

### 4. MIMIC-IV 2.1 pipeline (full 67 GB dataset — best results)

#### Prerequisites
1. Create a free account at https://www.kaggle.com
2. Go to **https://www.kaggle.com/settings** → API section → **"Create New Token"**
3. Save the downloaded `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`

#### Run
```bash
# Full pipeline (download → extract → train all models)
python src/run_mimiciv_pipeline.py

# Step-by-step (recommended for large dataset)
python src/data/download_mimiciv.py          # Download ZIP (~7.4 GB) from Kaggle
python src/data/explore_mimiciv.py           # Verify item IDs (< 30 sec)
python src/data/build_mimiciv_dataset.py     # Extract cardiac features (~30–90 min)
python src/utils/visualize_data.py
python src/models/train_lstm.py
python src/models/train_gru.py
python src/models/train_transformer.py
python src/utils/visualize_results.py

# Resume after download (if build already done)
python src/run_mimiciv_pipeline.py --skip_download --skip_build
```

#### Disk space required
| File | Size |
|------|------|
| MIMIC-IV ZIP (downloaded) | ~7.4 GB |
| Extracted cardiac features CSV | ~100–500 MB |
| Preprocessed .npy arrays | ~50–200 MB |
| **Total** | **~8–9 GB** |

---

## Project Structure
```
Temporal Analysis/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── data/
│   ├── mimic_demo/                        # Downloaded MIMIC-III Demo CSV files
│   ├── eicu_demo/                         # Downloaded eICU Demo CSV files
│   ├── real/                              # Fused real clinical CSV
│   ├── processed/                         # Synthetic longitudinal feature CSV
│   └── preprocessed/                      # .npy arrays for model training
├── models/                                # Saved model checkpoints (.pth)
├── src/
│   ├── __init__.py
│   ├── run_local_pipeline.py              # Synthetic data generation + preprocessing
│   ├── run_real_pipeline.py               # End-to-end real clinical pipeline orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_real_data.py          # Downloads MIMIC-III Demo + eICU Demo
│   │   ├── build_real_dataset.py          # Extracts features (MIMIC-III + eICU)
│   │   ├── download_mimiciv.py            # Downloads MIMIC-IV 2.1 ZIP from Kaggle
│   │   ├── explore_mimiciv.py             # Item ID discovery (d_items + d_labitems)
│   │   └── build_mimiciv_dataset.py       # Full MIMIC-IV extraction pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_lstm.py                  # BiLSTM with attention — primary model
│   │   ├── train_gru.py                   # GRU baseline model
│   │   ├── train_transformer.py           # Positional-encoded Transformer
│   │   └── predict.py                     # CLI inference on a single patient
│   └── utils/
│       ├── __init__.py
│       ├── visualize_data.py              # EDA plots (class balance, distributions, trends)
│       └── visualize_results.py           # Post-training plots (ROC, PR, threshold sweep)
└── visualizations/
    ├── eda/                               # EDA figures
    ├── lstm/                              # LSTM training history + confusion matrix
    └── results/                           # ROC curve, PR curve, threshold sweep
```

---

## File Descriptions

### Pipeline Runners
| File | Purpose |
|------|---------|
| `src/run_local_pipeline.py` | Generates synthetic longitudinal EHR data (58 features × 6 visits × 500 patients) |
| `src/run_real_pipeline.py` | MIMIC-III Demo + eICU Demo pipeline (open-access) |
| `src/run_mimiciv_pipeline.py` | **MIMIC-IV 2.1 full pipeline** — download → explore → extract → train |

### Data Scripts
| File | Purpose |
|------|---------|
| `src/data/download_real_data.py` | Downloads MIMIC-III Demo + eICU Demo from PhysioNet (no credentials) |
| `src/data/build_real_dataset.py` | Extracts cardiac features from MIMIC-III + eICU, fuses to 30 features |
| `src/data/download_mimiciv.py` | Downloads MIMIC-IV 2.1 ZIP from Kaggle (~7.4 GB) via Kaggle API |
| `src/data/explore_mimiciv.py` | Reads `d_items` + `d_labitems` from within ZIP to confirm itemid mappings |
| `src/data/build_mimiciv_dataset.py` | Streams `chartevents` + `labevents` in chunks from ZIP, builds 58-feature cardiac progression dataset |

### Model Training
| File | Purpose |
|------|---------|
| `src/models/train_lstm.py` | Trains a Bidirectional LSTM with self-attention head — the **primary model** |
| `src/models/train_gru.py` | Trains a Bidirectional GRU — lightweight baseline for comparison |
| `src/models/train_transformer.py` | Trains a Transformer with positional encoding and CLS-token classification |
| `src/models/predict.py` | CLI tool for single-patient inference using a trained LSTM checkpoint |

### Visualization
| File | Purpose |
|------|---------|
| `src/utils/visualize_data.py` | Generates EDA plots: class balance, feature distributions, temporal trends, missing values |
| `src/utils/visualize_results.py` | Generates post-training plots: ROC curve, Precision-Recall curve, threshold sweep |

---

## Models

| Model | Script | Architecture | Description |
|-------|--------|--------------|-------------|
| BiLSTM | `train_lstm.py` | 2-layer BiLSTM + Self-Attention + FC | Primary model with attention-weighted context |
| GRU | `train_gru.py` | 2-layer BiGRU + FC | Lightweight baseline using last hidden state |
| Transformer | `train_transformer.py` | Linear projection + Positional Encoding + 3-layer Encoder + CLS token | Modern attention-based architecture |

---

## Prediction Target — Cardiac Progression

For the MIMIC-IV pipeline, the model predicts whether a patient's cardiac
condition will **worsen (label=1)** or **improve / stay stable (label=0)** by
their next ICU admission.

This is determined by a **Cardiac Severity Index (CSI)** computed per ICU stay
from these components:

| Component | Normal Range | Weight |
|-----------|-------------|--------|
| Heart rate deviation | 60–100 bpm | up to +3 pts |
| Systolic BP (hypotension/hypertension) | 90–160 mmHg | up to +3 pts |
| Troponin T (myocardial injury) | < 0.01 ng/mL | up to +3 pts |
| BNP (heart failure severity) | < 100 pg/mL | up to +3 pts |
| SpO2 (oxygenation) | > 95% | up to +3 pts |
| Creatinine (cardiorenal) | < 1.5 mg/dL | up to +2 pts |
| Respiratory rate | 12–20 br/min | up to +2 pts |
| Hemoglobin (cardiac load) | > 12 g/dL | up to +2 pts |

If CSI increases > 10% from visit t to visit t+1 → **label = 1 (worsening)**.
For single-visit patients or last visit → hospital in-hospital mortality is used.

---

## Expected Performance

| Dataset | AUC-ROC | Notes |
|---------|---------|-------|
| Synthetic (500 pts) | 0.90–0.95 | Controlled noise, clean signal |
| MIMIC-III Demo (~100 ICU) | 0.65–0.75 | Very small sample |
| **MIMIC-IV Full (~300K visits)** | **0.75–0.85** | Best real-world performance |

---

## Dependencies

Listed in `requirements.txt`:
- **numpy** — array operations
- **pandas** — data manipulation
- **scikit-learn** — preprocessing, splitting, metrics
- **matplotlib + seaborn** — visualization
- **torch** — deep learning models
- **tqdm** — progress bars
- **requests** — HTTP downloads
- **joblib** — serialisation (scaler, feature names)
