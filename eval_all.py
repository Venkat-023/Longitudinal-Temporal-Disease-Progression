"""Quick eval script to collect metrics from all 3 models."""
import sys, numpy as np, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
sys.path.insert(0, "src/models")

X_test = np.load("data/preprocessed/X_test.npy")
y_test = np.load("data/preprocessed/y_test.npy")
Xt = torch.tensor(X_test, dtype=torch.float32)
device = torch.device("cpu")

def metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
    }

# --- LSTM ---
from train_lstm import LSTMHeartDiseaseModel
ckpt = torch.load("models/lstm_final.pth", map_location=device, weights_only=False)
lstm = LSTMHeartDiseaseModel(ckpt["input_size"], ckpt["hidden_size"], ckpt["num_layers"], ckpt["dropout"], True)
lstm.load_state_dict(ckpt["model_state_dict"])
lstm.eval()
with torch.no_grad():
    lstm_probs = torch.sigmoid(lstm(Xt)).numpy()
lstm_m = metrics(y_test, lstm_probs)

# --- GRU ---
from train_gru import GRUHeartDiseaseModel
gckpt = torch.load("models/gru_final.pth", map_location=device, weights_only=False)
gru = GRUHeartDiseaseModel(input_size=gckpt["input_size"])
gru.load_state_dict(gckpt["model_state_dict"])
gru.eval()
with torch.no_grad():
    gru_probs = torch.sigmoid(gru(Xt)).numpy()
gru_m = metrics(y_test, gru_probs)

# --- Transformer ---
from train_transformer import TransformerHeartDiseaseModel
tckpt = torch.load("models/transformer_final.pth", map_location=device, weights_only=False)
d_model, nhead = 128, 8
while d_model % nhead != 0:
    nhead -= 1
tf = TransformerHeartDiseaseModel(tckpt["input_size"], d_model, nhead, 3)
tf.load_state_dict(tckpt["model_state_dict"])
tf.eval()
with torch.no_grad():
    tf_probs = torch.sigmoid(tf(Xt)).numpy()
tf_m = metrics(y_test, tf_probs)

print("="*60)
print("  FINAL MODEL COMPARISON - Test Set")
print(f"  Test: {len(y_test)} samples, pos={int(y_test.sum())}, neg={int((y_test==0).sum())}")
print("="*60)
print(f"\n{'Metric':<12} {'LSTM':>10} {'GRU':>10} {'Transformer':>12}")
print("-"*48)
for k in ["Accuracy","Precision","Recall","F1","AUC-ROC"]:
    print(f"{k:<12} {lstm_m[k]:>10.4f} {gru_m[k]:>10.4f} {tf_m[k]:>12.4f}")
print("="*60)
