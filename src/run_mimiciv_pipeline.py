"""
run_mimiciv_pipeline.py
=======================
End-to-end MIMIC-IV 2.1 pipeline orchestrator.

Steps
-----
  1. Download MIMIC-IV 2.1 ZIP from Kaggle (~7.4 GB)
  2. Explore item IDs (optional sanity check)
  3. Build dataset from ZIP (streams from disk — does NOT unzip 67 GB)
  4. EDA visualisations
  5. Train BiLSTM / GRU / Transformer models
  6. Result plots (ROC, PR, threshold sweep)

Usage
-----
  # Full run (first-time setup)
  python src/run_mimiciv_pipeline.py

  # Skip download (ZIP already present)
  python src/run_mimiciv_pipeline.py --skip_download

  # Skip download + build (preprocessed .npy arrays already present)
  python src/run_mimiciv_pipeline.py --skip_download --skip_build

  # Train only specific models
  python src/run_mimiciv_pipeline.py --skip_download --skip_build --models lstm gru

Prerequisites
-------------
  1. pip install kaggle
  2. Set up Kaggle API key (see download_mimiciv.py instructions)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], desc: str, cwd: Path = ROOT):
    print(f"\n{'='*67}")
    print(f"  {desc}")
    print(f"{'='*67}")
    result = subprocess.run([sys.executable] + cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n❌  Step failed: {desc}")
        print("   Fix the error above and re-run with --skip_download "
              "and/or --skip_build to resume.")
        sys.exit(result.returncode)
    print(f"\n✅  {desc} — done")


def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-IV 2.1 end-to-end pipeline"
    )
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading (ZIP already present)")
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip dataset build (preprocessed .npy already exist)")
    parser.add_argument("--skip_explore", action="store_true", default=True,
                        help="Skip item ID exploration (default: skip for speed)")
    parser.add_argument("--models", nargs="+",
                        choices=["lstm", "gru", "transformer"],
                        default=["lstm", "gru", "transformer"],
                        help="Models to train (default: all three)")
    parser.add_argument("--chunk", type=int, default=100_000,
                        help="CSV chunk size for large files (default: 100,000)")
    parser.add_argument("--seq_len", type=int, default=6,
                        help="Sequence length in ICU stays (default: 6)")
    args = parser.parse_args()

    print("\n" + "🏥 " * 22)
    print("  MIMIC-IV 2.1 — Cardiac Progression Pipeline")
    print("  BiLSTM / GRU / Transformer  |  Progression prediction")
    print("🏥 " * 22)

    step = 1

    # ── Step 1: Download ───────────────────────────────────────────────────────
    if not args.skip_download:
        run(
            ["src/data/download_mimiciv.py"],
            f"Step {step}: Download MIMIC-IV 2.1 from Kaggle (~7.4 GB)"
        )
    else:
        print(f"\n⏭   Step {step}: Download  — skipped (--skip_download)")
    step += 1

    # ── Step 2: Explore (optional) ────────────────────────────────────────────
    if not args.skip_explore:
        run(
            ["src/data/explore_mimiciv.py"],
            f"Step {step}: Explore item IDs (d_items + d_labitems)"
        )
    else:
        print(f"\n⏭   Step {step}: Explore   — skipped (--skip_explore)")
    step += 1

    # ── Step 3: Build dataset ─────────────────────────────────────────────────
    if not args.skip_build:
        run(
            ["src/data/build_mimiciv_dataset.py",
             "--chunk", str(args.chunk),
             "--seq_len", str(args.seq_len)],
            f"Step {step}: Extract cardiac features & build sequences (~30-90 min)"
        )
    else:
        print(f"\n⏭   Step {step}: Build     — skipped (--skip_build)")
    step += 1

    # ── Step 4: EDA ───────────────────────────────────────────────────────────
    run(
        ["src/utils/visualize_data.py"],
        f"Step {step}: EDA visualisations"
    )
    step += 1

    # ── Step 5: Train models ──────────────────────────────────────────────────
    model_scripts = {
        "lstm":        "src/models/train_lstm.py",
        "gru":         "src/models/train_gru.py",
        "transformer": "src/models/train_transformer.py",
    }
    for model in args.models:
        run(
            [model_scripts[model]],
            f"Step {step}: Train {model.upper()}"
        )
        step += 1

    # ── Step 6: Result visualisations ────────────────────────────────────────
    run(
        ["src/utils/visualize_results.py"],
        f"Step {step}: Result plots (ROC, PR, threshold sweep)"
    )

    print("\n" + "✅ " * 22)
    print("  MIMIC-IV Pipeline complete!")
    print(f"  Models      : {ROOT / 'models'}")
    print(f"  Plots       : {ROOT / 'visualizations'}")
    print(f"  Preprocessed: {ROOT / 'data' / 'preprocessed'}")
    print("✅ " * 22)


if __name__ == "__main__":
    main()
