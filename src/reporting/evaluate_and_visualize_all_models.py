"""Evaluate and visualize all trained sequence models.

Works with both repository layouts:
    - cleaned layout: src/model_training/train_*.py
    - older Colab layout: src/models/train_*.py

Usage:
    python src/reporting/evaluate_and_visualize_all_models.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "visualizations" / "model_comparison"

for import_dir in [ROOT / "src" / "model_training", ROOT / "src" / "models"]:
    if import_dir.exists():
        sys.path.insert(0, str(import_dir))


def _load_class(module_names: list[str], class_name: str):
    last_error: Exception | None = None
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise ImportError(f"Could not import {class_name}: {last_error}")


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing test arrays in {DATA_DIR}")
    x_test = np.nan_to_num(np.load(x_path), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_test = np.load(y_path).astype(int)
    return x_test, y_test


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt
    return {"model_state_dict": ckpt}


def _build_lstm(ckpt: dict, input_size: int):
    cls = _load_class(["train_bilstm_attention", "train_lstm"], "LSTMHeartDiseaseModel")
    try:
        return cls(
            ckpt.get("input_size", input_size),
            ckpt.get("hidden_size", 128),
            ckpt.get("num_layers", 2),
            ckpt.get("dropout", 0.3),
            True,
        )
    except TypeError:
        return cls(
            input_size=ckpt.get("input_size", input_size),
            hidden_size=ckpt.get("hidden_size", 128),
            num_layers=ckpt.get("num_layers", 2),
            dropout=ckpt.get("dropout", 0.3),
        )


def _build_gru(ckpt: dict, input_size: int):
    cls = _load_class(["train_bigru", "train_gru"], "GRUHeartDiseaseModel")
    return cls(
        input_size=ckpt.get("input_size", input_size),
        hidden_size=ckpt.get("hidden_size", 128),
        num_layers=ckpt.get("num_layers", 2),
        dropout=ckpt.get("dropout", 0.3),
    )


def _build_transformer(ckpt: dict, input_size: int):
    cls = _load_class(["train_transformer_encoder", "train_transformer"], "TransformerHeartDiseaseModel")
    d_model = ckpt.get("d_model", 128)
    nhead = ckpt.get("nhead", 8)
    while d_model % nhead != 0 and nhead > 1:
        nhead -= 1
    return cls(
        input_size=ckpt.get("input_size", input_size),
        d_model=d_model,
        nhead=nhead,
        num_layers=ckpt.get("num_layers", 3),
        dropout=ckpt.get("dropout", 0.1),
    )


@torch.no_grad()
def _predict(model: torch.nn.Module, x_test: np.ndarray, device: torch.device) -> np.ndarray:
    model.to(device).eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    probs = torch.sigmoid(model(x_tensor)).detach().cpu().numpy()
    return probs.reshape(-1)


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | list[list[int]]]:
    preds = (probs >= 0.5).astype(int)
    roc = roc_auc_score(y_true, probs) if len(set(y_true.tolist())) > 1 else math.nan
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "auc_roc": float(roc),
        "confusion_matrix": confusion_matrix(y_true, preds).astype(int).tolist(),
    }


def _plot_metric_bars(results: dict[str, dict], out_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    names = list(results)
    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, metric in enumerate(metrics):
        values = [results[name][metric] for name in names]
        ax.bar(x + (idx - 2) * width, values, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Held-Out Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_metrics.png", dpi=160)
    plt.close(fig)


def _plot_roc_pr(y_true: np.ndarray, probabilities: dict[str, np.ndarray], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ax.plot(recall, precision, linewidth=2, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_curves.png", dpi=160)
    plt.close(fig)


def _plot_confusion_matrices(results: dict[str, dict], out_dir: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["Stable", "Worse"])
        ax.set_yticks([0, 1], ["Stable", "Worse"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_test, y_test = _load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = {
        "BiLSTM Attention": ("lstm_final.pth", _build_lstm),
        "BiGRU": ("gru_final.pth", _build_gru),
        "Transformer Encoder": ("transformer_final.pth", _build_transformer),
    }

    results: dict[str, dict] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, (filename, builder) in specs.items():
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"Skipping {name}: missing {path}")
            continue
        ckpt = _load_checkpoint(path, device)
        model = builder(ckpt, x_test.shape[2])
        model.load_state_dict(ckpt["model_state_dict"])
        probs = _predict(model, x_test, device)
        probabilities[name] = probs
        results[name] = _metrics(y_test, probs)

    if not results:
        raise FileNotFoundError(f"No trained checkpoints found in {MODEL_DIR}")

    output = {
        "test_size": int(len(y_test)),
        "positive": int(y_test.sum()),
        "negative": int((y_test == 0).sum()),
        "models": results,
    }
    (ROOT / "results_all_models.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    _plot_metric_bars(results, OUT_DIR)
    _plot_roc_pr(y_test, probabilities, OUT_DIR)
    _plot_confusion_matrices(results, OUT_DIR)

    print("\nFinal Model Comparison - Test Set")
    print(f"{'Model':<24} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 72)
    for name, row in results.items():
        print(
            f"{name:<24} {row['accuracy']:>8.4f} {row['precision']:>8.4f} "
            f"{row['recall']:>8.4f} {row['f1_score']:>8.4f} {row['auc_roc']:>8.4f}"
        )
    print(f"\nSaved JSON: {ROOT / 'results_all_models.json'}")
    print(f"Saved plots: {OUT_DIR}")


if __name__ == "__main__":
    main()
