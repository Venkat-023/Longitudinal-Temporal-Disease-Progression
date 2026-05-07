"""
train_bigru.py
==============
Bidirectional GRU baseline model for heart disease prediction.
Mirrors the LSTM training script but uses a GRU encoder.

Usage
-----
python src/model_training/train_bigru.py [--epochs 50] [--lr 0.001]
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
VIZ_DIR   = ROOT / "visualizations" / "gru"


class GRUHeartDiseaseModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        # Use last hidden state (both directions concatenated)
        context = out[:, -1, :]
        return self.fc(context).squeeze(-1)


def load_data(data_dir: Path):
    arrays = {}
    for name in ["X_train","X_val","X_test","y_train","y_val","y_test"]:
        p = data_dir / f"{name}.npy"
        if not p.exists():
            print(f"❌ Missing: {p}")
            print("   Run: python src/run_local_pipeline.py")
            sys.exit(1)
        arrays[name] = np.load(p)
    return arrays


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += ((torch.sigmoid(logits) > 0.5).long() == yb.long()).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--hidden_size",type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=10)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  GRU Heart Disease Prediction Model")
    print("=" * 70)

    data = load_data(DATA_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(data["X_train"], data["y_train"], args.batch_size)
    val_loader   = make_loader(data["X_val"],   data["y_val"],   args.batch_size, shuffle=False)
    test_loader  = make_loader(data["X_test"],  data["y_test"],  args.batch_size, shuffle=False)

    input_size = data["X_train"].shape[2]
    model = GRUHeartDiseaseModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"\n🧠 GRU Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    pos_weight = torch.tensor(
        [(data["y_train"] == 0).sum() / max((data["y_train"] == 1).sum(), 1)]
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience_ctr = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {device}\n")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        print(f"Epoch {epoch:>3}/{args.epochs}  Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_DIR / "gru_best.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Evaluate
    model.load_state_dict(torch.load(MODEL_DIR / "gru_best.pth", map_location=device))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_probs.extend(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
            all_labels.extend(yb.numpy())
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    print(f"\n📊 GRU Test Metrics:")
    print(f"   Accuracy  : {accuracy_score(labels, preds):.4f}")
    print(f"   Precision : {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"   Recall    : {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"   F1 Score  : {f1_score(labels, preds, zero_division=0):.4f}")
    try:
        print(f"   AUC-ROC   : {roc_auc_score(labels, probs):.4f}")
    except Exception:
        pass

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }, MODEL_DIR / "gru_final.pth")
    print(f"\n💾 Model saved to: {MODEL_DIR / 'gru_final.pth'}")
    print("\n✅ GRU training complete!")


if __name__ == "__main__":
    main()
