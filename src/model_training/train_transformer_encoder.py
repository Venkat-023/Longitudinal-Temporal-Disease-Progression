"""
train_transformer_encoder.py
============================
Positional-encoding Transformer classifier for heart disease prediction.

Usage
-----
python src/model_training/train_transformer_encoder.py [--epochs 50] [--lr 0.0001]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerHeartDiseaseModel(nn.Module):
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        x = self.input_proj(x)                          # (batch, seq, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls, x], dim=1)                  # (batch, seq+1, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.fc(x[:, 0, :]).squeeze(-1)           # CLS token output


def load_data(data_dir: Path):
    arrays = {}
    for name in ["X_train","X_val","X_test","y_train","y_val","y_test"]:
        p = data_dir / f"{name}.npy"
        if not p.exists():
            print(f"❌ Missing: {p}. Run: python src/run_local_pipeline.py")
            sys.exit(1)
        arrays[name] = np.load(p)
    return arrays


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
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
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--num_layers", type=int,   default=3)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--patience",   type=int,   default=10)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Transformer Heart Disease Prediction Model")
    print("=" * 70)

    data = load_data(DATA_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure nhead divides d_model
    while args.d_model % args.nhead != 0:
        args.nhead -= 1

    input_size = data["X_train"].shape[2]
    model = TransformerHeartDiseaseModel(
        input_size, args.d_model, args.nhead, args.num_layers, args.dropout
    ).to(device)
    print(f"\n🧠 Transformer: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}")

    train_loader = make_loader(data["X_train"], data["y_train"], args.batch_size)
    val_loader   = make_loader(data["X_val"],   data["y_val"],   args.batch_size, False)
    test_loader  = make_loader(data["X_test"],  data["y_test"],  args.batch_size, False)

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
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   optimizer, criterion, device, False)
        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_DIR / "transformer_best.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "transformer_best.pth", map_location=device))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_probs.extend(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
            all_labels.extend(yb.numpy())
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    print(f"\n📊 Transformer Test Metrics:")
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
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }, MODEL_DIR / "transformer_final.pth")
    print(f"\n💾 Model saved to: {MODEL_DIR / 'transformer_final.pth'}")
    print("\n✅ Transformer training complete!")


if __name__ == "__main__":
    main()
