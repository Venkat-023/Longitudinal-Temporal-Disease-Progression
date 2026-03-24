"""
run_local_pipeline.py
=====================
Synthetic data generator + end-to-end pipeline runner.

Usage
-----
python src/run_local_pipeline.py [--num_patients 500] [--num_timepoints 6]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PROCESSED_DIR = DATA_DIR / "processed"

FEATURE_NAMES = [
    # Vital signs
    "heart_rate_mean", "heart_rate_std",
    "sbp_mean", "sbp_std",
    "dbp_mean", "dbp_std",
    "mbp_mean", "mbp_std",
    "resp_rate_mean", "resp_rate_std",
    "temperature_mean", "temperature_std",
    "spo2_mean", "spo2_std",
    "glucose_fingerstick_mean", "glucose_fingerstick_std",
    # Cardiac markers
    "troponin_t_mean", "troponin_t_max",
    "troponin_i_mean", "troponin_i_max",
    "bnp_mean", "bnp_max",
    "nt_probnp_mean", "nt_probnp_max",
    # Metabolic
    "creatinine_mean", "creatinine_max",
    "bun_mean", "bun_max",
    "glucose_lab_mean", "glucose_lab_std",
    # Electrolytes
    "potassium_mean", "potassium_std",
    "sodium_mean", "sodium_std",
    "chloride_mean", "chloride_std",
    "bicarbonate_mean", "bicarbonate_std",
    # CBC
    "hematocrit_mean", "hematocrit_std",
    "hemoglobin_mean", "hemoglobin_std",
    "wbc_mean", "wbc_max",
    "platelets_mean", "platelets_std",
    # Clinical context
    "age", "los_days", "num_procedures",
    "prior_mi", "prior_hf", "prior_arrhythmia",
    "num_medications", "icu_flag",
    # Admission-level
    "admission_type_emergency", "admission_type_elective",
    "days_since_last_admission",
    # Temporal features (added last)
    "time_delta_days", "visit_index",
]

NUM_FEATURES = len(FEATURE_NAMES)   # 58 features


def _disease_signal(age, prior_mi, prior_hf, t):
    """Monotonically increasing risk score used to generate labels."""
    base = 0.1 + 0.003 * age + 0.25 * prior_mi + 0.30 * prior_hf + 0.05 * t
    return base


def generate_synthetic_data(num_patients: int = 500, num_timepoints: int = 6,
                             seed: int = 42) -> tuple:
    """
    Generate synthetic longitudinal EHR data.

    Returns
    -------
    X : ndarray of shape (num_patients, num_timepoints, num_features)
    y : ndarray of shape (num_patients,)  — binary label at LAST visit
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print("  Generating Synthetic Longitudinal EHR Data")
    print(f"{'='*60}")
    print(f"  Patients    : {num_patients}")
    print(f"  Time-points : {num_timepoints}")
    print(f"  Features    : {NUM_FEATURES}")
    print(f"{'='*60}\n")

    X = np.zeros((num_patients, num_timepoints, NUM_FEATURES))
    y = np.zeros(num_patients, dtype=np.int64)

    feat = {name: i for i, name in enumerate(FEATURE_NAMES)}

    for p in range(num_patients):
        age = rng.integers(40, 90)
        prior_mi = int(rng.random() < 0.3)
        prior_hf = int(rng.random() < 0.25)
        prior_arr = int(rng.random() < 0.35)

        for t in range(num_timepoints):
            noise = lambda s: rng.normal(0, s)

            risk = _disease_signal(age, prior_mi, prior_hf, t)

            # Vitals — deteriorate with risk
            hr = 72 + 15 * risk + noise(8)
            X[p, t, feat["heart_rate_mean"]] = hr
            X[p, t, feat["heart_rate_std"]] = abs(noise(5))
            X[p, t, feat["sbp_mean"]] = 120 + 20 * risk + noise(12)
            X[p, t, feat["sbp_std"]] = abs(noise(8))
            X[p, t, feat["dbp_mean"]] = 80 + 10 * risk + noise(8)
            X[p, t, feat["dbp_std"]] = abs(noise(5))
            X[p, t, feat["mbp_mean"]] = 93 + 13 * risk + noise(8)
            X[p, t, feat["mbp_std"]] = abs(noise(5))
            X[p, t, feat["resp_rate_mean"]] = 16 + 4 * risk + noise(2)
            X[p, t, feat["resp_rate_std"]] = abs(noise(1.5))
            X[p, t, feat["temperature_mean"]] = 37.0 + 0.5 * risk + noise(0.3)
            X[p, t, feat["temperature_std"]] = abs(noise(0.2))
            X[p, t, feat["spo2_mean"]] = max(88, 98 - 5 * risk + noise(1))
            X[p, t, feat["spo2_std"]] = abs(noise(1))
            X[p, t, feat["glucose_fingerstick_mean"]] = 100 + 40 * risk + noise(15)
            X[p, t, feat["glucose_fingerstick_std"]] = abs(noise(10))

            # Cardiac markers
            X[p, t, feat["troponin_t_mean"]] = max(0, 0.01 + 0.5 * risk + noise(0.1))
            X[p, t, feat["troponin_t_max"]] = max(0, 0.02 + 0.8 * risk + noise(0.2))
            X[p, t, feat["troponin_i_mean"]] = max(0, 0.1 + 5 * risk + noise(1))
            X[p, t, feat["troponin_i_max"]] = max(0, 0.2 + 8 * risk + noise(2))
            X[p, t, feat["bnp_mean"]] = max(0, 50 + 300 * risk + noise(50))
            X[p, t, feat["bnp_max"]] = max(0, 80 + 500 * risk + noise(80))
            X[p, t, feat["nt_probnp_mean"]] = max(0, 100 + 600 * risk + noise(100))
            X[p, t, feat["nt_probnp_max"]] = max(0, 200 + 900 * risk + noise(150))

            # Metabolic
            X[p, t, feat["creatinine_mean"]] = max(0.4, 1.0 + 1.5 * risk + noise(0.3))
            X[p, t, feat["creatinine_max"]] = max(0.4, 1.2 + 2 * risk + noise(0.5))
            X[p, t, feat["bun_mean"]] = max(5, 15 + 30 * risk + noise(5))
            X[p, t, feat["bun_max"]] = max(5, 18 + 40 * risk + noise(8))
            X[p, t, feat["glucose_lab_mean"]] = 100 + 40 * risk + noise(15)
            X[p, t, feat["glucose_lab_std"]] = abs(noise(10))

            # Electrolytes
            X[p, t, feat["potassium_mean"]] = 4.0 + 0.5 * risk + noise(0.3)
            X[p, t, feat["potassium_std"]] = abs(noise(0.2))
            X[p, t, feat["sodium_mean"]] = 140 - 3 * risk + noise(2)
            X[p, t, feat["sodium_std"]] = abs(noise(1))
            X[p, t, feat["chloride_mean"]] = 102 - 2 * risk + noise(2)
            X[p, t, feat["chloride_std"]] = abs(noise(1))
            X[p, t, feat["bicarbonate_mean"]] = 24 - 3 * risk + noise(2)
            X[p, t, feat["bicarbonate_std"]] = abs(noise(1.5))

            # CBC
            X[p, t, feat["hematocrit_mean"]] = max(20, 40 - 8 * risk + noise(3))
            X[p, t, feat["hematocrit_std"]] = abs(noise(2))
            X[p, t, feat["hemoglobin_mean"]] = max(7, 13 - 3 * risk + noise(1))
            X[p, t, feat["hemoglobin_std"]] = abs(noise(0.8))
            X[p, t, feat["wbc_mean"]] = max(2, 7 + 5 * risk + noise(2))
            X[p, t, feat["wbc_max"]] = max(2, 9 + 8 * risk + noise(3))
            X[p, t, feat["platelets_mean"]] = max(50, 230 - 50 * risk + noise(30))
            X[p, t, feat["platelets_std"]] = abs(noise(20))

            # Clinical context
            X[p, t, feat["age"]] = age
            X[p, t, feat["los_days"]] = max(1, 3 + 5 * risk + noise(2))
            X[p, t, feat["num_procedures"]] = max(0, round(2 + 4 * risk + noise(1)))
            X[p, t, feat["prior_mi"]] = prior_mi
            X[p, t, feat["prior_hf"]] = prior_hf
            X[p, t, feat["prior_arrhythmia"]] = prior_arr
            X[p, t, feat["num_medications"]] = max(1, round(5 + 8 * risk + noise(2)))
            X[p, t, feat["icu_flag"]] = int(risk > 0.6)

            # Admission type
            X[p, t, feat["admission_type_emergency"]] = int(risk > 0.5)
            X[p, t, feat["admission_type_elective"]] = int(risk <= 0.5)
            X[p, t, feat["days_since_last_admission"]] = max(0, 180 - 40 * risk + noise(30)) if t > 0 else 0

            # Temporal
            X[p, t, feat["time_delta_days"]] = t * 60 + noise(10)
            X[p, t, feat["visit_index"]] = t

        # Label: high-risk at final visit → positive
        final_risk = _disease_signal(age, prior_mi, prior_hf, num_timepoints - 1)
        prob = 1 / (1 + np.exp(-5 * (final_risk - 0.55)))
        y[p] = int(rng.random() < prob)

    pos_rate = y.mean()
    print(f"  ✅ Generated {num_patients} patients × {num_timepoints} visits × {NUM_FEATURES} features")
    print(f"  Label balance: {y.sum()} positive ({pos_rate:.1%}), {(1-y).sum()} negative ({1-pos_rate:.1%})\n")

    return X, y


def preprocess_and_save(X: np.ndarray, y: np.ndarray):
    """Split, scale, and save arrays to data/preprocessed/."""
    n = len(y)

    # 70 / 15 / 15 split
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15 / 0.85,
                                                        random_state=42, stratify=y_tv)

    # Fit scaler on training data only (flatten → fit → reshape)
    n_train, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(n_train, T, F)
    X_val_scaled = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PREPROCESSED_DIR / "X_train.npy", X_train_scaled)
    np.save(PREPROCESSED_DIR / "X_val.npy",   X_val_scaled)
    np.save(PREPROCESSED_DIR / "X_test.npy",  X_test_scaled)
    np.save(PREPROCESSED_DIR / "y_train.npy", y_train)
    np.save(PREPROCESSED_DIR / "y_val.npy",   y_val)
    np.save(PREPROCESSED_DIR / "y_test.npy",  y_test)
    joblib.dump(scaler, PREPROCESSED_DIR / "preprocessor.pkl")

    print("📊 Final Dataset Shapes:")
    print(f"   X_train : {X_train_scaled.shape}, y_train : {y_train.shape}")
    print(f"   X_val   : {X_val_scaled.shape},   y_val   : {y_val.shape}")
    print(f"   X_test  : {X_test_scaled.shape},  y_test  : {y_test.shape}")
    print(f"\n💾 Saved preprocessed arrays to: {PREPROCESSED_DIR}")

    # Save a CSV summary for visualisation scripts
    rows = []
    for split, Xs, ys in [("train", X_train_scaled, y_train),
                            ("val",   X_val_scaled,   y_val),
                            ("test",  X_test_scaled,  y_test)]:
        for i in range(len(ys)):
            for t in range(Xs.shape[1]):
                row = {"split": split, "patient": i, "timepoint": t, "label": ys[i]}
                row.update({FEATURE_NAMES[f]: Xs[i, t, f] for f in range(len(FEATURE_NAMES))})
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_DIR / "synthetic_longitudinal_features.csv", index=False)
    print(f"💾 Saved longitudinal CSV to: {PROCESSED_DIR / 'synthetic_longitudinal_features.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data and run pipeline")
    parser.add_argument("--num_patients", type=int, default=500)
    parser.add_argument("--num_timepoints", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = generate_synthetic_data(args.num_patients, args.num_timepoints, args.seed)
    preprocess_and_save(X, y)

    print("\n✅ Synthetic pipeline complete!")
    print("   Next steps:")
    print("   1. python src/utils/visualize_data.py")
    print("   2. python src/models/train_lstm.py")


if __name__ == "__main__":
    main()
