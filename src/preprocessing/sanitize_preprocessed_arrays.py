"""Repair saved preprocessed arrays by replacing NaN/Inf values with 0.

Use this when the dataset build completed but training starts with NaN loss:

    python src/preprocessing/sanitize_preprocessed_arrays.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "preprocessed"


def sanitize_x(path: Path) -> None:
    arr = np.load(path)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"{path.name}: shape={arr.shape}, nan={n_nan}, inf={n_inf}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.save(path, arr)


def sanitize_y(path: Path) -> None:
    arr = np.load(path).astype(np.int64)
    bad = set(np.unique(arr).tolist()) - {0, 1}
    print(f"{path.name}: shape={arr.shape}, labels={sorted(np.unique(arr).tolist())}")
    if bad:
        raise ValueError(f"{path.name} contains non-binary labels: {sorted(bad)}")
    np.save(path, arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize preprocessed MIMIC-IV .npy arrays")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    required = [f"{kind}_{split}.npy" for kind in ["X", "y"] for split in ["train", "val", "test"]]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        print(f"Missing arrays in {data_dir}: {missing}")
        sys.exit(1)

    for split in ["train", "val", "test"]:
        sanitize_x(data_dir / f"X_{split}.npy")
        sanitize_y(data_dir / f"y_{split}.npy")

    print(f"\nSanitized arrays saved in: {data_dir}")


if __name__ == "__main__":
    main()
