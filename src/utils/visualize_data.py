"""
visualize_data.py
=================
Exploratory Data Analysis (EDA) plots for the synthetic / MIMIC-III dataset.

Usage
-----
python src/utils/visualize_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT     = Path(__file__).resolve().parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"
PREP_DIR = ROOT / "data" / "preprocessed"
OUT_DIR  = ROOT / "visualizations" / "eda"

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f78166"
PURPLE   = "#d2a8ff"
YELLOW   = "#e3b341"


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


def plot_class_balance(y_train, y_val, y_test, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor(DARK_BG)
    splits = [("Train", y_train), ("Val", y_val), ("Test", y_test)]
    colors = [[GREEN, RED], [GREEN, RED], [GREEN, RED]]

    for ax, (name, y), cols in zip(axes, splits, colors):
        counts = [int((y == 0).sum()), int((y == 1).sum())]
        bars = ax.bar(["Healthy", "Disease"], counts, color=cols, width=0.5,
                      edgecolor=BORDER, linewidth=0.8)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{count}\n({count/len(y):.0%})",
                    ha="center", va="bottom", fontsize=9.5, color="white")
        ax.set_title(f"{name} Split", fontsize=12, fontweight="bold")
        ax.set_ylabel("Patients")
        _apply_dark(ax)

    fig.suptitle("Class Balance per Split", fontsize=14, fontweight="bold",
                 color="white", y=1.02)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "class_balance.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"   ✅ {out_dir / 'class_balance.png'}")


def plot_feature_distributions(X_train: np.ndarray, out_dir: Path,
                               feature_names: list[str] | None = None):
    T, F = X_train.shape[1], X_train.shape[2]
    n_show = min(12, F)
    indices = list(range(n_show))

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = axes.flatten()

    for idx, feat_i in enumerate(indices):
        ax = axes_flat[idx]
        vals = X_train[:, :, feat_i].flatten()
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=30, color=BLUE, alpha=0.8, edgecolor=BORDER, linewidth=0.5)
        label = feature_names[feat_i] if feature_names else f"Feature {feat_i}"
        ax.set_title(label, fontsize=9, fontweight="bold")
        _apply_dark(ax)

    for ax in axes_flat[n_show:]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions (Training Set, First 12 Features)",
                 fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_distributions.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"   ✅ {out_dir / 'feature_distributions.png'}")


def plot_temporal_trends(X_train: np.ndarray, y_train: np.ndarray,
                         out_dir: Path, feature_names: list[str] | None = None):
    T = X_train.shape[1]
    n_feats = min(6, X_train.shape[2])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = axes.flatten()

    for idx in range(n_feats):
        ax = axes_flat[idx]
        for label_val, color, name in [(0, GREEN, "Healthy"), (1, RED, "Disease")]:
            mask = y_train == label_val
            means = X_train[mask, :, idx].mean(axis=0)
            stds  = X_train[mask, :, idx].std(axis=0)
            ts = np.arange(T)
            ax.plot(ts, means, color=color, linewidth=2, label=name)
            ax.fill_between(ts, means - stds, means + stds, color=color, alpha=0.2)

        label = feature_names[idx] if feature_names else f"Feature {idx}"
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Visit")
        ax.set_ylabel("Normalised Value")
        ax.legend(facecolor="#21262d", labelcolor="white", fontsize=8, framealpha=0.8)
        _apply_dark(ax)

    for ax in axes_flat[n_feats:]:
        ax.set_visible(False)

    fig.suptitle("Temporal Trends: Healthy vs Disease Patients",
                 fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(out_dir / "temporal_trends.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"   ✅ {out_dir / 'temporal_trends.png'}")


def plot_missing_values(X_train: np.ndarray, out_dir: Path,
                         feature_names: list[str] | None = None):
    miss_rate = np.isnan(X_train).mean(axis=(0, 1))  # per feature
    n = len(miss_rate)
    if miss_rate.max() == 0:
        print("   ℹ️  No missing values in training set — skipping missing value plot.")
        return

    sorted_idx = np.argsort(miss_rate)[::-1][:30]
    labels = [feature_names[i] if feature_names else f"F{i}" for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.barh(labels, miss_rate[sorted_idx] * 100, color=YELLOW, edgecolor=BORDER,
            linewidth=0.5)
    ax.set_xlabel("Missing Rate (%)")
    ax.set_title("Top Features by Missing Value Rate", fontsize=12, fontweight="bold")
    _apply_dark(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "missing_values.png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"   ✅ {out_dir / 'missing_values.png'}")


def main():
    print("\n" + "=" * 60)
    print("  Exploratory Data Analysis")
    print("=" * 60)

    # Try to load preprocessed arrays
    for fname in ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                  "X_test.npy", "y_test.npy"]:
        if not (PREP_DIR / fname).exists():
            print(f"❌ Missing: {PREP_DIR / fname}")
            print("   Run: python src/run_local_pipeline.py first.")
            return

    X_train = np.load(PREP_DIR / "X_train.npy")
    y_train = np.load(PREP_DIR / "y_train.npy")
    X_val   = np.load(PREP_DIR / "X_val.npy")
    y_val   = np.load(PREP_DIR / "y_val.npy")
    X_test  = np.load(PREP_DIR / "X_test.npy")
    y_test  = np.load(PREP_DIR / "y_test.npy")

    print(f"\n📊 Dataset Overview:")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_val   : {X_val.shape}")
    print(f"   X_test  : {X_test.shape}")

    # Feature names (optional — from preprocessed pkl)
    feature_names = None
    try:
        import joblib
        feat_path = PREP_DIR / "feature_names.pkl"
        if feat_path.exists():
            feature_names = joblib.load(feat_path)
    except Exception:
        pass

    print("\n📊 Generating plots...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_class_balance(y_train, y_val, y_test, OUT_DIR)
    plot_feature_distributions(X_train, OUT_DIR, feature_names)
    plot_temporal_trends(X_train, y_train, OUT_DIR, feature_names)
    plot_missing_values(X_train, OUT_DIR, feature_names)

    print(f"\n✅ All EDA plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
