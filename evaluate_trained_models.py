"""Evaluate all trained sequence models on the held-out test set.

Run after training:
    python evaluate_trained_models.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "preprocessed"
sys.path.insert(0, str(ROOT / "src" / "model_training"))


def _load_test_data() -> tuple[np.ndarray, np.ndarray]:
    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Missing test arrays. Build the dataset first with "
            "python src/preprocessing/build_cardiac_progression_dataset.py"
        )
    return np.load(x_path), np.load(y_path)


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_true, probs) if len(set(y_true.tolist())) > 1 else math.nan
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auc_roc": auc,
    }


@torch.no_grad()
def _predict(model: torch.nn.Module, x_test: np.ndarray) -> np.ndarray:
    model.eval()
    xt = torch.tensor(x_test, dtype=torch.float32)
    return torch.sigmoid(model(xt)).cpu().numpy()


def _eval_lstm(x_test: np.ndarray) -> np.ndarray | None:
    path = MODEL_DIR / "lstm_final.pth"
    if not path.exists():
        return None
    from train_bilstm_attention import LSTMHeartDiseaseModel

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LSTMHeartDiseaseModel(
        ckpt["input_size"],
        ckpt.get("hidden_size", 128),
        ckpt.get("num_layers", 2),
        ckpt.get("dropout", 0.3),
        True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return _predict(model, x_test)


def _eval_gru(x_test: np.ndarray) -> np.ndarray | None:
    path = MODEL_DIR / "gru_final.pth"
    if not path.exists():
        return None
    from train_bigru import GRUHeartDiseaseModel

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GRUHeartDiseaseModel(
        input_size=ckpt["input_size"],
        hidden_size=ckpt.get("hidden_size", 128),
        num_layers=ckpt.get("num_layers", 2),
        dropout=ckpt.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return _predict(model, x_test)


def _eval_transformer(x_test: np.ndarray) -> np.ndarray | None:
    path = MODEL_DIR / "transformer_final.pth"
    if not path.exists():
        return None
    from train_transformer_encoder import TransformerHeartDiseaseModel

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    d_model, nhead = ckpt.get("d_model", 128), ckpt.get("nhead", 8)
    while d_model % nhead != 0:
        nhead -= 1
    model = TransformerHeartDiseaseModel(
        ckpt["input_size"],
        d_model,
        nhead,
        ckpt.get("num_layers", 3),
        ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return _predict(model, x_test)


def main() -> None:
    x_test, y_test = _load_test_data()
    evaluators = {
        "bilstm_attention": _eval_lstm,
        "bigru": _eval_gru,
        "transformer": _eval_transformer,
    }

    results: dict[str, object] = {
        "test_size": int(len(y_test)),
        "positive": int(y_test.sum()),
        "negative": int((y_test == 0).sum()),
        "models": {},
    }

    for name, evaluator in evaluators.items():
        probs = evaluator(x_test)
        if probs is None:
            print(f"Skipping {name}: checkpoint not found")
            continue
        results["models"][name] = _metrics(y_test, probs)

    print("=" * 72)
    print("  Final Model Comparison - Test Set")
    print(f"  Test: {len(y_test)} samples, pos={int(y_test.sum())}, neg={int((y_test == 0).sum())}")
    print("=" * 72)
    print(f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 72)
    for name, metrics in results["models"].items():
        print(
            f"{name:<20} "
            f"{metrics['accuracy']:>8.4f} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['auc_roc']:>8.4f}"
        )
    print("=" * 72)

    out_path = ROOT / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
