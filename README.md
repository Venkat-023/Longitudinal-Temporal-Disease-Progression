# Heart Disease Temporal Analysis — Deep Learning Pipeline

A longitudinal deep-learning pipeline that predicts heart disease progression
using **Bidirectional LSTM with attention**, **GRU**, and **Transformer** models,
trained on either **synthetic** or **real clinical data** (MIMIC-III Demo + eICU Demo).

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

### 3. Real clinical data pipeline (open-access — no credentials needed)
```bash
python src/run_real_pipeline.py
```
Or run individual steps:
```bash
python src/data/download_real_data.py      # Download MIMIC-III Demo + eICU Demo
python src/data/build_real_dataset.py      # Feature extraction + fusion + preprocessing
python src/utils/visualize_data.py         # EDA visualizations
python src/models/train_lstm.py            # Train LSTM model
python src/models/train_gru.py             # Train GRU model
python src/models/train_transformer.py     # Train Transformer model
python src/utils/visualize_results.py      # ROC, PR, threshold sweep
```

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
│   │   └── build_real_dataset.py          # Extracts features, fuses, preprocesses
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
| `src/run_local_pipeline.py` | Generates synthetic longitudinal EHR data (58 features × 6 visits × 500 patients), splits/scales/saves to `.npy` arrays |
| `src/run_real_pipeline.py` | Orchestrates the full real-data pipeline: download → build → EDA → train all models → result plots |

### Data Scripts
| File | Purpose |
|------|---------|
| `src/data/download_real_data.py` | Auto-downloads MIMIC-III Demo (100 ICU patients) and eICU Demo (~2,500 unit stays) from PhysioNet — no credentials required |
| `src/data/build_real_dataset.py` | Extracts cardiac cohorts from both databases, aggregates vitals + labs per ICU stay, fuses into 30 shared features, builds sliding-window sequences, and saves preprocessed `.npy` arrays |

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

## Expected Performance (Synthetic Data)

| Metric    | Expected Range |
|-----------|---------------|
| Accuracy  | 0.85 – 0.92   |
| Precision | 0.83 – 0.90   |
| Recall    | 0.84 – 0.91   |
| F1        | 0.84 – 0.90   |
| AUC-ROC   | 0.90 – 0.95   |

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
