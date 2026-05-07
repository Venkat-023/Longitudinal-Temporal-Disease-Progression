"""
predict_bilstm_patient.py
==========
CLI inference script: load a trained LSTM and predict on a patient.

Usage
-----
python src/model_training/predict_bilstm_patient.py --patient_id 0
python src/model_training/predict_bilstm_patient.py --patient_id 42 --model_path models/lstm_final.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models"


def load_model(model_path: Path, input_size: int, device: torch.device):
    """Load the saved LSTM model checkpoint."""
    sys.path.insert(0, str(ROOT / "src" / "models"))
    from train_bilstm_attention import LSTMHeartDiseaseModel

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = {k: ckpt.get(k, v) for k, v in
               [("hidden_size", 128), ("num_layers", 2), ("dropout", 0.3)]}
        model = LSTMHeartDiseaseModel(
            input_size=ckpt.get("input_size", input_size),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=True,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        if "metrics" in ckpt:
            print(f"   (Model trained with AUC-ROC: {ckpt['metrics'].get('auc_roc', 'N/A'):.4f})")
    else:
        # Bare state dict
        model = LSTMHeartDiseaseModel(input_size=input_size, bidirectional=True)
        model.load_state_dict(ckpt)

    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Heart disease prediction inference")
    parser.add_argument("--patient_id", type=int, default=0,
                        help="Index of patient in test set")
    parser.add_argument("--model_path", type=str,
                        default=str(MODEL_DIR / "lstm_final.pth"),
                        help="Path to saved .pth model file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Heart Disease Prediction — Inference")
    print("=" * 60)

    # ── Load test data ────────────────────────────────────────────
    for fname in ["X_test.npy", "y_test.npy"]:
        if not (DATA_DIR / fname).exists():
            print(f"❌ Missing: {DATA_DIR / fname}")
            print("   Run: python src/run_local_pipeline.py first.")
            sys.exit(1)

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    if args.patient_id >= len(y_test):
        print(f"❌ patient_id {args.patient_id} out of range (max: {len(y_test)-1})")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python src/model_training/train_bilstm_attention.py first.")
        sys.exit(1)

    # ── Setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_test.shape[2]

    print(f"\n📂 Test set: {len(y_test)} patients")
    print(f"🔍 Predicting patient index: {args.patient_id}")

    model = load_model(model_path, input_size, device)

    # ── Inference ─────────────────────────────────────────────────
    x = torch.tensor(X_test[[args.patient_id]], dtype=torch.float32).to(device)
    with torch.no_grad():
        logit = model(x)
        prob  = torch.sigmoid(logit).item()

    label = int(y_test[args.patient_id])
    pred  = int(prob >= args.threshold)

    print(f"\n{'─'*50}")
    print(f"  Patient ID        : {args.patient_id}")
    print(f"  Sequence shape    : {X_test[args.patient_id].shape}")
    print(f"  Disease probability: {prob:.4f} ({prob*100:.1f}%)")
    print(f"  Predicted label   : {'⚠️  DISEASE' if pred else '✅ HEALTHY'}")
    print(f"  Actual label      : {'⚠️  DISEASE' if label else '✅ HEALTHY'}")
    print(f"  Correct?          : {'✅ Yes' if pred == label else '❌ No'}")
    print(f"{'─'*50}")

    # ── Batch evaluation ──────────────────────────────────────────
    print(f"\n📊 Running on all {len(y_test)} test patients...")
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        all_probs = torch.sigmoid(model(X_t.to(device))).cpu().numpy()

    all_preds = (all_probs >= args.threshold).astype(int)
    correct = (all_preds == y_test).sum()
    print(f"   Overall accuracy: {correct}/{len(y_test)} = {correct/len(y_test):.1%}")


if __name__ == "__main__":
    main()
