"""
plot_model_results.py
=====================
Post-training result visualisations: ROC curve, PR curve, feature importance.

Usage
-----
python src/reporting/plot_model_results.py [--model_path models/lstm_final.pth]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

ROOT     = Path(__file__).resolve().parent.parent.parent
OUT_DIR  = ROOT / "visualizations" / "results"
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models"

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f78166"
PURPLE   = "#d2a8ff"


def _apply_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def plot_roc(labels, probs, out_dir: Path, model_name: str = "LSTM"):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.5,
            label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=BORDER, linestyle="--", linewidth=1.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#21262d", labelcolor="white", framealpha=0.8)
    _apply_dark(ax)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"   ✅ {out_dir / 'roc_curve.png'}  (AUC={roc_auc:.4f})")


def plot_pr_curve(labels, probs, out_dir: Path, model_name: str = "LSTM"):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(recall, precision, color=GREEN, linewidth=2.5,
            label=f"{model_name} (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#21262d", labelcolor="white", framealpha=0.8)
    _apply_dark(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"   ✅ {out_dir / 'pr_curve.png'}  (AP={ap:.4f})")


def plot_threshold_sweep(labels, probs, out_dir: Path):
    """Plot F1, Precision, Recall for thresholds 0.1–0.9."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    f1s, precs, recs = [], [], []
    from sklearn.metrics import f1_score, precision_score, recall_score
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1s.append(f1_score(labels, preds, zero_division=0))
        precs.append(precision_score(labels, preds, zero_division=0))
        recs.append(recall_score(labels, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(thresholds, f1s,   color=BLUE,   linewidth=2, label="F1")
    ax.plot(thresholds, precs, color=GREEN,  linewidth=2, label="Precision", linestyle="--")
    ax.plot(thresholds, recs,  color=RED,    linewidth=2, label="Recall", linestyle="-.")
    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(best_t, color=PURPLE, linestyle=":", linewidth=1.5,
               label=f"Best threshold ({best_t:.2f})")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#21262d", labelcolor="white", framealpha=0.8)
    _apply_dark(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"   ✅ {out_dir / 'threshold_sweep.png'}  (Best threshold: {best_t:.2f})")


def get_probabilities(model_path: Path, X_test: np.ndarray) -> np.ndarray:
    """Load model and run inference on the test set."""
    sys.path.insert(0, str(ROOT / "src" / "model_training"))
    from train_bilstm_attention import LSTMHeartDiseaseModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_test.shape[2]
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        hidden = ckpt.get("hidden_size", 128)
        layers = ckpt.get("num_layers", 2)
        drop   = ckpt.get("dropout", 0.3)
        model = LSTMHeartDiseaseModel(input_size, hidden, layers, drop, True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = LSTMHeartDiseaseModel(input_size, 128, 2, 0.3, True)
        model.load_state_dict(ckpt)

    model.to(device).eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(X_t)).cpu().numpy()
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default=str(MODEL_DIR / "lstm_final.pth"))
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Post-Training Result Visualisations")
    print("=" * 60)

    for fname in ["X_test.npy", "y_test.npy"]:
        if not (DATA_DIR / fname).exists():
            print(f"❌ Missing: {DATA_DIR / fname}")
            print("   Run: python src/preprocessing/build_cardiac_progression_dataset.py first.")
            return

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python src/model_training/train_bilstm_attention.py first.")
        return

    print(f"\n📂 Loading model from {model_path.name}...")
    probs = get_probabilities(model_path, X_test)

    print("\n📊 Generating result visualisations...")
    plot_roc(y_test, probs, OUT_DIR)
    plot_pr_curve(y_test, probs, OUT_DIR)
    plot_threshold_sweep(y_test, probs, OUT_DIR)

    print(f"\n✅ All result plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
