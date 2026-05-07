"""
train_bilstm_attention.py
=========================
Trains a Bidirectional LSTM with self-attention for heart disease prediction.
Works with both synthetic and MIMIC-III preprocessed data.

Usage
-----
python src/model_training/train_bilstm_attention.py [--epochs 50] [--lr 0.001] [--batch_size 32]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models"
VIZ_DIR   = ROOT / "visualizations" / "lstm"


# ─────────────────────────── Model Architecture ────────────────────────────── #

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attention(lstm_out).squeeze(-1)     # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)         # (batch, hidden)
        return context


class LSTMHeartDiseaseModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.attention = AttentionLayer(self.hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)          # (batch, seq, hidden)
        context = self.attention(lstm_out)   # (batch, hidden)
        return self.fc(context).squeeze(-1)  # (batch,)


# ─────────────────────────── Training Helpers ──────────────────────────────── #

def load_data(data_dir: Path) -> dict:
    missing = [f for f in ["X_train.npy","X_val.npy","X_test.npy",
                            "y_train.npy","y_val.npy","y_test.npy"]
               if not (data_dir / f).exists()]
    if missing:
        print(f"❌ Missing files: {missing}")
        print("   Run: python src/run_local_pipeline.py   (synthetic)")
        print("   OR:  python src/preprocessing/build_cardiac_progression_dataset.py")
        sys.exit(1)

    return {
        "X_train": np.load(data_dir / "X_train.npy"),
        "X_val":   np.load(data_dir / "X_val.npy"),
        "X_test":  np.load(data_dir / "X_test.npy"),
        "y_train": np.load(data_dir / "y_train.npy"),
        "y_val":   np.load(data_dir / "y_val.npy"),
        "y_test":  np.load(data_dir / "y_test.npy"),
    }


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y_batch.long()).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y_batch.long()).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        probs = torch.sigmoid(model(X_batch)).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(y_batch.numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


# ─────────────────────────── Visualisation ─────────────────────────────────── #

def plot_training_history(history: dict, save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], color="#58a6ff", linewidth=2, label="Train")
    axes[0].plot(epochs, history["val_loss"],   color="#f78166", linewidth=2, label="Val",
                 linestyle="--")
    axes[0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend(facecolor="#21262d", labelcolor="white", framealpha=0.8)
    if "best_epoch" in history:
        axes[0].axvline(history["best_epoch"], color="#3fb950", linestyle=":", alpha=0.7,
                        label=f"Best epoch ({history['best_epoch']})")

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], color="#58a6ff", linewidth=2, label="Train")
    axes[1].plot(epochs, history["val_acc"],   color="#f78166", linewidth=2, label="Val",
                 linestyle="--")
    axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(facecolor="#21262d", labelcolor="white", framealpha=0.8)

    fig.suptitle("LSTM Training History", fontsize=16, fontweight="bold", color="white")
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "training_history.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📊 Saved: {save_dir / 'training_history.png'}")


def plot_confusion_matrix(cm: np.ndarray, save_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Disease"],
                yticklabels=["Healthy", "Disease"],
                ax=ax, linewidths=0.5, linecolor="#30363d",
                annot_kws={"size": 14, "color": "white"})

    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white")

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📊 Saved: {save_dir / 'confusion_matrix.png'}")


# ─────────────────────────── Main ──────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM for heart disease prediction")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--hidden_size", type=int,   default=128)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int,   default=10)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  LSTM Heart Disease Prediction Model")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────
    print("\n📂 Loading data...")
    data = load_data(DATA_DIR)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    print(f"✅ Data loaded:")
    print(f"   Train : {X_train.shape}, Labels: {y_train.shape}")
    print(f"   Val   : {X_val.shape},   Labels: {y_val.shape}")
    print(f"   Test  : {X_test.shape},  Labels: {y_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────
    model = LSTMHeartDiseaseModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Architecture:\n{model}")
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable:,}")

    # Handle class imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                               dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ── Training ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Training LSTM Model")
    print(f"{'=' * 70}")
    print(f"Device         : {device}")
    print(f"Epochs         : {args.epochs}")
    print(f"Learning rate  : {args.lr}")
    print(f"Patience       : {args.patience}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_ctr = 0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f} | Val Acc: {vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    history["best_epoch"] = best_epoch
    print(f"\n✅ Training complete!")

    # ── Evaluation ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Test Set Evaluation")
    print(f"{'=' * 70}")

    # Load best checkpoint
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))
    probs, labels = predict(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(labels, preds)

    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    # ── Save final model ──────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec,
                    "f1": f1, "auc_roc": auc},
    }, MODEL_DIR / "lstm_final.pth")
    print(f"\n💾 Model saved to: {MODEL_DIR / 'lstm_final.pth'}")

    # ── Plots ─────────────────────────────────────────────────────
    print("\n📊 Generating visualizations...")
    plot_training_history(history, VIZ_DIR)
    plot_confusion_matrix(cm, VIZ_DIR)

    print(f"\n✅ LSTM training complete!")


if __name__ == "__main__":
    main()
