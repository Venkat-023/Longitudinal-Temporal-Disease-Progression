"""
run_real_pipeline.py
====================
Single-command orchestrator: downloads real clinical databases,
builds the dataset, and trains all three models.

Usage
-----
python src/run_real_pipeline.py [--skip_download] [--skip_build]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], desc: str):
    print(f"\n{'='*65}")
    print(f"  {desc}")
    print(f"{'='*65}")
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n❌ Step failed: {desc}")
        print("   Fix the error above, then re-run with --skip_download and/or "
              "--skip_build to resume from a later step.")
        sys.exit(result.returncode)
    print(f"\n✅ {desc} — done")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end real clinical data pipeline"
    )
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading data (already downloaded)")
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip building dataset (already preprocessed)")
    parser.add_argument("--models", nargs="+",
                        choices=["lstm", "gru", "transformer"],
                        default=["lstm", "gru", "transformer"],
                        help="Which models to train (default: all)")
    args = parser.parse_args()

    print("\n" + "🏥 " * 21)
    print("  REAL CLINICAL DATA PIPELINE")
    print("  MIMIC-III Demo + eICU Demo → LSTM / GRU / Transformer")
    print("🏥 " * 21)

    # ── Step 1: Download ───────────────────────────────────────────
    if not args.skip_download:
        run(
            ["src/data/download_real_data.py"],
            "Step 1/5: Downloading MIMIC-III Demo + eICU Demo"
        )
    else:
        print("\n⏭  Skipping download (--skip_download)")

    # ── Step 2: Build dataset ──────────────────────────────────────
    if not args.skip_build:
        run(
            ["src/data/build_real_dataset.py"],
            "Step 2/5: Feature extraction + fusion + preprocessing"
        )
    else:
        print("\n⏭  Skipping dataset build (--skip_build)")

    # ── Step 3: EDA visualisation ──────────────────────────────────
    run(
        ["src/utils/visualize_data.py"],
        "Step 3/5: EDA visualisation"
    )

    # ── Step 4: Train models ───────────────────────────────────────
    model_scripts = {
        "lstm":        "src/models/train_lstm.py",
        "gru":         "src/models/train_gru.py",
        "transformer": "src/models/train_transformer.py",
    }
    for i, model in enumerate(args.models, start=4):
        run(
            [model_scripts[model]],
            f"Step {i}/{3+len(args.models)}: Training {model.upper()} model"
        )

    # ── Step 5: Result visualisation ──────────────────────────────
    run(
        ["src/utils/visualize_results.py"],
        f"Step {3+len(args.models)+1}: Result visualisations (ROC, PR, threshold)"
    )

    print("\n" + "✅ " * 21)
    print("  ALL DONE! Real clinical data pipeline complete.")
    print(f"  Models saved to   : {ROOT / 'models'}")
    print(f"  Visualisations    : {ROOT / 'visualizations'}")
    print(f"  Preprocessed data : {ROOT / 'data' / 'preprocessed'}")
    print("✅ " * 21)


if __name__ == "__main__":
    main()
