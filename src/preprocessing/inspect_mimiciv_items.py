"""
inspect_mimiciv_items.py
==================
Fast discovery script — runs in < 30 seconds.
Reads only the tiny dictionary files (d_items.csv, d_labitems.csv) from inside
the ZIP to confirm which itemid values map to our 58 target features.
Run this BEFORE build_cardiac_progression_dataset.py.

Usage
-----
python src/preprocessing/inspect_mimiciv_items.py [--zip data/mimic_iv_raw/mimic-iv-2-1.zip]
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent.parent
ZIP_PATH = ROOT / "data" / "mimic_iv_raw" / "mimic-iv-2-1.zip"


# ─── Target feature → search keywords in d_items / d_labitems label column ───
ICU_VITAL_KEYWORDS = {
    "heart_rate":   ["heart rate"],
    "sbp":          ["arterial blood pressure systolic", "non invasive blood pressure s", "blood pressure systolic"],
    "dbp":          ["arterial blood pressure diastolic", "non invasive blood pressure d", "blood pressure diastolic"],
    "mbp":          ["arterial blood pressure mean", "non invasive blood pressure m", "blood pressure mean"],
    "resp_rate":    ["respiratory rate"],
    "temperature":  ["temperature fahrenheit", "temperature celsius"],
    "spo2":         ["o2 saturation pulseoxymetry", "spo2", "oxygen saturation"],
    "glucose_fs":   ["glucose finger stick", "fingerstick glucose", "glucose"],
}

HOSP_LAB_KEYWORDS = {
    "troponin_t":   ["troponin t"],
    "troponin_i":   ["troponin i"],
    "bnp":          ["b-type natriuretic peptide", "bnp"],
    "nt_probnp":    ["nt-probnp", "n-terminal", "proBNP"],
    "creatinine":   ["creatinine"],
    "bun":          ["urea nitrogen"],
    "potassium":    ["potassium"],
    "sodium":       ["sodium"],
    "chloride":     ["chloride"],
    "bicarbonate":  ["bicarbonate"],
    "hematocrit":   ["hematocrit"],
    "hemoglobin":   ["hemoglobin"],
    "wbc":          ["white blood cells", "leukocytes"],
    "platelets":    ["platelet count", "platelets"],
    "glucose_lab":  ["glucose"],
}


def _open_csv_from_zip(zf: zipfile.ZipFile, pattern: str) -> pd.DataFrame | None:
    """Find and read a CSV from the ZIP using fuzzy path matching."""
    # Try exact match first
    names = zf.namelist()
    match = next((n for n in names if n.endswith(pattern)), None)
    if match is None:
        # Some Kaggle uploads flatten directories
        base = pattern.split("/")[-1]
        match = next((n for n in names if n.endswith(base)), None)
    if match is None:
        return None
    with zf.open(match) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), low_memory=False)


def _search_items(df: pd.DataFrame, label_col: str, keywords: list[str]) -> list[tuple]:
    """Return (itemid, label) rows matching any keyword (case-insensitive)."""
    mask = df[label_col].str.lower().str.contains(
        "|".join([k.lower() for k in keywords]), na=False
    )
    return list(df[mask][["itemid", label_col]].itertuples(index=False, name=None))


def explore(zip_path: Path):
    print(f"\n{'='*65}")
    print("  MIMIC-IV Item ID Discovery")
    print(f"  ZIP: {zip_path}")
    print(f"{'='*65}")

    if not zip_path.exists():
        print(f"\n❌  ZIP not found: {zip_path}")
        print("    Run: python src/preprocessing/download_mimiciv_dataset.py")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zf:

        # ── 1. d_items (ICU vitals dictionary) ────────────────────────────────
        print("\n── ICU Vitals (chartevents) ───────────────────────────────")
        d_items = _open_csv_from_zip(zf, "icu/d_items.csv")
        if d_items is None:
            print("   ⚠️  d_items.csv not found in ZIP — will use fallback itemids")
        else:
            d_items.columns = [c.lower() for c in d_items.columns]
            label_col = next(c for c in d_items.columns if "label" in c)
            print(f"   d_items rows: {len(d_items):,}  |  label column: '{label_col}'\n")
            print(f"   {'Feature':<20} {'itemid(s)':<30} {'Label'}")
            print(f"   {'-'*70}")
            for feat, keywords in ICU_VITAL_KEYWORDS.items():
                hits = _search_items(d_items, label_col, keywords)
                if hits:
                    ids  = [str(h[0]) for h in hits[:4]]
                    lbl  = hits[0][1][:35]
                    print(f"   {feat:<20} {', '.join(ids):<30} {lbl}")
                else:
                    print(f"   {feat:<20} {'(not found)':<30}")

        # ── 2. d_labitems (hosp labs dictionary) ──────────────────────────────
        print("\n── Hospital Labs (labevents) ──────────────────────────────")
        d_lab = _open_csv_from_zip(zf, "hosp/d_labitems.csv")
        if d_lab is None:
            print("   ⚠️  d_labitems.csv not found in ZIP — will use fallback itemids")
        else:
            d_lab.columns = [c.lower() for c in d_lab.columns]
            label_col = next(c for c in d_lab.columns if "label" in c)
            print(f"   d_labitems rows: {len(d_lab):,}  |  label column: '{label_col}'\n")
            print(f"   {'Feature':<20} {'itemid(s)':<30} {'Label'}")
            print(f"   {'-'*70}")
            for feat, keywords in HOSP_LAB_KEYWORDS.items():
                hits = _search_items(d_lab, label_col, keywords)
                if hits:
                    ids  = [str(h[0]) for h in hits[:4]]
                    lbl  = hits[0][1][:35]
                    print(f"   {feat:<20} {', '.join(ids):<30} {lbl}")
                else:
                    print(f"   {feat:<20} {'(not found)':<30}")

        # ── 3. Show ZIP file listing ───────────────────────────────────────────
        print("\n── ZIP Contents ───────────────────────────────────────────")
        all_names = sorted(zf.namelist())
        csv_names = [n for n in all_names if n.endswith(".csv")]
        print(f"   Total CSV files in ZIP: {len(csv_names)}")
        for n in csv_names:
            info = zf.getinfo(n)
            size_mb = info.compress_size / 1e6
            print(f"   {n:<45} {size_mb:>8.1f} MB  (compressed)")

    print(f"\n{'='*65}")
    print("  ✅ Exploration complete!")
    print("  Next step:")
    print("  python src/preprocessing/build_cardiac_progression_dataset.py")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Explore MIMIC-IV ZIP item IDs")
    parser.add_argument("--zip", type=Path, default=ZIP_PATH,
                        help="Path to the MIMIC-IV 2.1 ZIP file")
    args = parser.parse_args()
    explore(args.zip)


if __name__ == "__main__":
    main()
