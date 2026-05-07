"""Full MIMIC-IV preprocessing and GPU training runner for Google Colab.

This script is intentionally strict: by default it rebuilds the preprocessed
arrays from the MIMIC-IV ZIP so final training does not accidentally use the
small smoke-test arrays left in data/preprocessed.

Example:
    python src/run_full_colab_training.py --zip data/mimic_iv_raw/mimic-iv-2-1.zip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualizations"


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {desc}")
    print(f"{'=' * 78}")
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def remove_generated_outputs() -> None:
    for path in [PREP_DIR, MODEL_DIR, VIZ_DIR, ROOT / "results.json"]:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Removed generated output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess full MIMIC-IV data and train models on Colab GPU"
    )
    parser.add_argument("--zip", type=Path, default=ROOT / "data" / "mimic_iv_raw" / "mimic-iv-2-1.zip")
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--skip_build", action="store_true", help="Reuse existing preprocessed arrays")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow running without CUDA")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bilstm_attention", "bigru", "transformer_encoder"],
        default=["bilstm_attention", "bigru", "transformer_encoder"],
    )
    args = parser.parse_args()

    zip_path = args.zip if args.zip.is_absolute() else ROOT / args.zip
    if not zip_path.exists():
        raise FileNotFoundError(f"MIMIC-IV ZIP not found: {zip_path}")

    cuda_ok = torch.cuda.is_available()
    print("\nColab/GPU check")
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif not args.allow_cpu:
        raise RuntimeError("CUDA is not available. In Colab, set Runtime -> Change runtime type -> GPU.")

    if not args.skip_build:
        remove_generated_outputs()
        run(
            [
                "src/preprocessing/build_cardiac_progression_dataset.py",
                "--zip",
                str(zip_path),
                "--chunk",
                str(args.chunk),
                "--seq_len",
                str(args.seq_len),
            ],
            "Build complete MIMIC-IV cardiac progression arrays",
        )

    model_commands = {
        "bilstm_attention": [
            "src/model_training/train_bilstm_attention.py",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--hidden_size",
            str(args.hidden_size),
            "--patience",
            str(args.patience),
        ],
        "bigru": [
            "src/model_training/train_bigru.py",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--hidden_size",
            str(args.hidden_size),
            "--patience",
            str(args.patience),
        ],
        "transformer_encoder": [
            "src/model_training/train_transformer_encoder.py",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--d_model",
            str(args.hidden_size),
            "--nhead",
            "8",
            "--num_layers",
            "3",
            "--patience",
            str(args.patience),
        ],
    }

    for model_name in args.models:
        run(model_commands[model_name], f"Train {model_name.upper()} on GPU")

    run(["evaluate_trained_models.py"], "Compare trained models on held-out test set")
    run(["src/reporting/plot_model_results.py"], "Generate result visualizations")

    print("\nDone. Key outputs:")
    print(f"Models: {MODEL_DIR}")
    print(f"Metrics: {ROOT / 'results.json'}")
    print(f"Plots: {VIZ_DIR}")


if __name__ == "__main__":
    main()
