# Model Results

This page records the latest full MIMIC-IV Colab result reported from the GPU
run on the generated `(sequence_length=6, features=59)` arrays.

## Dataset Used

| Split | Samples | Positives | Positive Rate |
|---|---:|---:|---:|
| Train | 24,678 | 4,178 | 16.93% |
| Validation | 5,289 | 896 | 16.94% |
| Test | 5,289 | 896 | 16.94% |

The first training attempt produced `NaN` losses because the saved `X_*.npy`
arrays still contained missing values after preprocessing and scaling:

| Split | NaN Values Replaced | Infinite Values |
|---|---:|---:|
| Train | 296,136 | 0 |
| Validation | 63,468 | 0 |
| Test | 63,468 | 0 |

The existing arrays were sanitized by replacing non-finite feature values with
zero before restarting training. After that repair, the LSTM/BiLSTM attention
model trained normally.

## BiLSTM Attention Result

Training command used in Colab:

```bash
python src/models/train_lstm.py --epochs 80 --lr 0.0003 --batch_size 128 --hidden_size 128 --patience 12
```

Training summary:

| Setting | Value |
|---|---:|
| GPU | Tesla T4 |
| Epochs requested | 80 |
| Early stopping epoch | 17 |
| Best epoch | 5 |
| Best validation loss | 0.6628 |
| Learning rate | 0.0003 |
| Batch size | 128 |
| Hidden size | 128 |
| Patience | 12 |

Test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.8134 |
| Precision | 0.4712 |
| Recall | 0.8304 |
| F1 score | 0.6012 |
| AUC-ROC | 0.9021 |

Confusion matrix:

| | Predicted Stable/Improving | Predicted Worsening |
|---|---:|---:|
| Actual Stable/Improving | 3,558 | 835 |
| Actual Worsening | 152 | 744 |

Generated outputs:

```text
models/lstm_final.pth
visualizations/lstm/training_history.png
visualizations/lstm/confusion_matrix.png
```

## Why These Results Improved

The first run was not valid because `NaN` feature values caused `NaN` losses.
The apparent 83% validation accuracy in that failed run was just the model
predicting the majority stable/improving class. Since only about 16.9% of
samples are worsening, a model can look accurate while missing nearly all
worsening cases.

After sanitization, training became meaningful. The strong AUC-ROC of `0.9021`
shows that the model ranks worsening cases well across thresholds. The recall
of `0.8304` means it identified most worsening cases on the test set. Precision
is lower at `0.4712` because the default `0.5` threshold produces false
positives, which is common when optimizing for high recall on an imbalanced
clinical task.

## Methods That Produced The Result

The result came from these methods:

- A cardiac MIMIC-IV cohort selected from diagnosis codes.
- ICU/admission timelines converted into fixed-length sequences of 6 visits.
- A 59-feature representation containing vitals, cardiac labs, renal/metabolic
  labs, CBC features, clinical context, and temporal gaps.
- A Cardiac Severity Index based label that separates worsening progression
  from stable or improving progression.
- Feature imputation, standardization, and post-scaling finite-value
  sanitization.
- A 2-layer bidirectional LSTM encoder with attention pooling over timesteps.
- Weighted `BCEWithLogitsLoss` to handle the 16.9% positive-class imbalance.
- AdamW optimization, dropout, gradient clipping, validation-loss checkpointing,
  and early stopping.

The most important interpretation is that the model traded some accuracy and
precision for much better worsening-case recall. For this task, AUC-ROC, recall,
and F1 are more informative than accuracy alone.
