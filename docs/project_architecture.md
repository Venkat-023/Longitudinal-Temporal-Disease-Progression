# Project Architecture

This is the current clean project map. If Colab training errors, use this file
to know exactly where to look.

## Top-Level Layout

```text
Temporal Analysis/
|-- README.md
|-- requirements.txt
|-- evaluate_trained_models.py
|-- docs/
|-- notebooks/
|-- data/
|-- src/
|-- models/              # generated after training
|-- visualizations/      # generated after plotting
`-- results.json         # generated after evaluation
```

## Important Top-Level Files

| File | Purpose |
|---|---|
| `README.md` | Main project instructions and commands |
| `requirements.txt` | Python packages needed in Colab/local runs |
| `evaluate_trained_models.py` | Loads trained checkpoints and compares all models |

## Documentation

```text
docs/
|-- project_architecture.md
|-- dataset.md
|-- model_results.md
|-- colab_vscode.md
`-- models/
    |-- bilstm_attention.md
    |-- bigru.md
    `-- transformer_encoder.md
```

| File | Purpose |
|---|---|
| `docs/dataset.md` | Raw dataset, extraction, preprocessing, labels, final features |
| `docs/model_results.md` | Latest Colab metrics, confusion matrix, and result interpretation |
| `docs/models/bilstm_attention.md` | BiLSTM attention architecture and training details |
| `docs/models/bigru.md` | BiGRU architecture and training details |
| `docs/models/transformer_encoder.md` | Transformer encoder architecture and training details |
| `docs/colab_vscode.md` | Colab and VS Code workflow notes |

## Notebook

```text
notebooks/
`-- mimiciv_colab_gpu_training.ipynb
```

Use this in Google Colab. It mounts Drive, installs dependencies, checks GPU,
and runs the full Colab training pipeline.

If the notebook cannot find the project folder, fix the `PROJECT_DIR` cell.

## Data Folders

```text
data/
|-- mimic_iv_raw/
|   `-- mimic-iv-2-1.zip
|-- real/
|   `-- mimiciv_longitudinal_features.csv
`-- preprocessed/
    |-- X_train.npy
    |-- X_val.npy
    |-- X_test.npy
    |-- y_train.npy
    |-- y_val.npy
    |-- y_test.npy
    |-- preprocessor.pkl
    `-- feature_names.pkl
```

| Folder | Purpose |
|---|---|
| `data/mimic_iv_raw/` | Raw MIMIC-IV ZIP lives here |
| `data/real/` | Generated longitudinal feature CSV |
| `data/preprocessed/` | Generated model-ready `.npy` arrays |

Do not edit generated arrays manually. If they are wrong, rerun preprocessing.

## Source Code Layout

```text
src/
|-- run_full_colab_training.py
|-- run_full_local_pipeline.py
|-- preprocessing/
|-- model_training/
`-- reporting/
```

## Full Pipeline Runners

### `src/run_full_colab_training.py`

Best entry point for Colab GPU.

It does:

1. Check CUDA/GPU.
2. Optionally remove old generated outputs.
3. Build complete cardiac progression dataset.
4. Train BiLSTM attention, BiGRU, and Transformer encoder.
5. Evaluate all models.
6. Generate result plots.

If the full Colab command fails, the console header tells you which step failed.

### `src/run_full_local_pipeline.py`

Local orchestrator for the same general workflow. Colab should usually use
`src/run_full_colab_training.py`.

## Preprocessing Code

```text
src/preprocessing/
|-- build_cardiac_progression_dataset.py
|-- download_mimiciv_dataset.py
|-- inspect_mimiciv_items.py
|-- sanitize_preprocessed_arrays.py
`-- __init__.py
```

| File | Purpose |
|---|---|
| `build_cardiac_progression_dataset.py` | Main dataset builder from MIMIC-IV ZIP to `.npy` arrays |
| `download_mimiciv_dataset.py` | Kaggle download helper |
| `inspect_mimiciv_items.py` | Inspect MIMIC item IDs for vitals/labs |
| `sanitize_preprocessed_arrays.py` | Repairs existing `X_*.npy` arrays if they contain NaN/Inf values |

Inside `build_cardiac_progression_dataset.py`, common fixes are:

| Error Area | Function To Inspect |
|---|---|
| cohort loading | `load_cohort` |
| vitals / `chartevents` | `extract_vitals` |
| labs / `labevents` | `extract_labs` |
| CSI and labels | `merge_and_label` |
| `.npy` arrays and scaling | `build_and_save` |

## Model Training Code

```text
src/model_training/
|-- train_bilstm_attention.py
|-- train_bigru.py
|-- train_transformer_encoder.py
|-- predict_bilstm_patient.py
|-- data_checks.py
`-- __init__.py
```

| File | Purpose |
|---|---|
| `train_bilstm_attention.py` | Trains 2-layer BiLSTM with attention |
| `train_bigru.py` | Trains 2-layer BiGRU baseline |
| `train_transformer_encoder.py` | Trains Transformer encoder classifier |
| `predict_bilstm_patient.py` | Runs prediction with trained BiLSTM checkpoint |
| `data_checks.py` | Validates saved arrays before training so NaN/Inf values fail early |

Generated checkpoints:

```text
models/lstm_final.pth
models/gru_final.pth
models/transformer_final.pth
```

## Reporting Code

```text
src/reporting/
|-- plot_dataset_overview.py
|-- plot_model_results.py
`-- __init__.py
```

| File | Purpose |
|---|---|
| `plot_dataset_overview.py` | Creates class balance, feature distribution, and temporal trend plots |
| `plot_model_results.py` | Creates ROC, PR curve, and threshold sweep plots |

## Quick Error Lookup

| Symptom | Where To Look |
|---|---|
| ZIP missing | `data/mimic_iv_raw/mimic-iv-2-1.zip` |
| Drive path error in Colab | `notebooks/mimiciv_colab_gpu_training.ipynb` |
| GPU unavailable | Colab runtime settings and `src/run_full_colab_training.py` |
| Cohort variable/name error | `src/preprocessing/build_cardiac_progression_dataset.py` |
| Long `chartevents` step | normal; in `extract_vitals` |
| Lab extraction error | `extract_labs` in preprocessing file |
| Missing `X_train.npy` | preprocessing did not complete |
| `Train Loss: nan` | check `data/preprocessed/X_*.npy`; rebuild or sanitize arrays |
| BiLSTM error | `src/model_training/train_bilstm_attention.py` |
| BiGRU error | `src/model_training/train_bigru.py` |
| Transformer error | `src/model_training/train_transformer_encoder.py` |
| Evaluation error | `evaluate_trained_models.py` |
| Plot error | `src/reporting/plot_model_results.py` |

## Main Colab Command

```bash
python src/run_full_colab_training.py --zip data/mimic_iv_raw/mimic-iv-2-1.zip --chunk 100000 --seq_len 6 --epochs 80 --batch_size 128 --hidden_size 128 --patience 12
```
