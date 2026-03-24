"""
download_real_data.py
=====================
Auto-downloads two freely available (no credentials) real clinical databases:

  1. MIMIC-III Clinical Database Demo v1.4  — 100 ICU patients, full MIMIC schema
     Source: https://physionet.org/content/mimiciii-demo/1.4/
     License: Open Data Commons Open Database License (ODbL)

  2. eICU Collaborative Research Database Demo v2.0.1 — ~2,500 ICU unit stays
     Source: https://physionet.org/content/eicu-crd-demo/2.0.1/
     License: Open Data Commons Open Database License (ODbL)

Usage
-----
python src/data/download_real_data.py
"""

import gzip
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parent.parent.parent
MIMIC_DIR = ROOT / "data" / "mimic_demo"
EICU_DIR  = ROOT / "data" / "eicu_demo"

# ── MIMIC-III Demo — download as a single ZIP (13.4 MB, fast) ─────────────────
MIMIC_ZIP_URL = "https://physionet.org/content/mimiciii-demo/get-zip/1.4/"

# ── eICU Demo (gzipped CSV) ───────────────────────────────────────────────────
EICU_BASE = "https://physionet.org/files/eicu-crd-demo/2.0.1"
EICU_FILES = [
    "patient.csv.gz",
    "vitalPeriodic.csv.gz",
    "vitalAperiodic.csv.gz",
    "lab.csv.gz",
    "diagnosis.csv.gz",
    "apachePatientResult.csv.gz",
]


def _download_bytes(url: str, desc: str) -> bytes:
    """Download a URL into memory with progress indicator."""
    print(f"   Downloading {desc}...", end="", flush=True)
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (research-pipeline; heart-disease-lstm)"
    })
    with urlopen(req, timeout=600) as resp:
        data = resp.read()
    size_mb = len(data) / 1e6
    print(f"  ✅ ({size_mb:.1f} MB)")
    return data


def _download_file(url: str, dest: Path):
    """Stream-download a file to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"   ⏭  {dest.name} already exists — skipping.")
        return

    print(f"   Downloading {dest.name}...", end="", flush=True)
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (research-pipeline; heart-disease-lstm)"
    })
    try:
        with urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        size_mb = dest.stat().st_size / 1e6
        print(f"  ✅ ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        if dest.exists():
            dest.unlink()


def _decompress_gz(gz_path: Path) -> Path:
    """Decompress a .gz file and return path to decompressed file."""
    out_path = gz_path.with_suffix("")  # strip .gz
    if out_path.exists():
        return out_path
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"   📦 Decompressed → {out_path.name}")
    return out_path


def download_mimic_demo():
    """Download MIMIC-III Demo as a single ZIP archive and extract."""
    print("\n" + "─" * 60)
    print("  MIMIC-III Clinical Database Demo  (100 ICU patients)")
    print("─" * 60)
    MIMIC_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if (MIMIC_DIR / "PATIENTS.csv").exists() and (MIMIC_DIR / "CHARTEVENTS.csv").exists():
        print("   ⏭  Already downloaded — skipping.")
        return

    # Download ZIP
    zip_data = _download_bytes(MIMIC_ZIP_URL, "MIMIC-III Demo ZIP (13.4 MB)")

    # Extract
    print("   📦 Extracting...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for member in zf.namelist():
            # Files are in mimiciii-demo-1.4/  subfolder inside the ZIP
            basename = os.path.basename(member)
            if basename.endswith(".csv") or basename.endswith(".txt"):
                dest = MIMIC_DIR / basename
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"      ✅ {basename}")

    print(f"\n   📂 Files saved to: {MIMIC_DIR}")


def download_eicu_demo():
    """Download eICU Demo gzipped CSV files."""
    print("\n" + "─" * 60)
    print("  eICU Collaborative Research Database Demo  (~2,500 stays)")
    print("─" * 60)
    EICU_DIR.mkdir(parents=True, exist_ok=True)

    for fname in EICU_FILES:
        url  = f"{EICU_BASE}/{fname}"
        dest = EICU_DIR / fname
        _download_file(url, dest)

        if dest.exists() and fname.endswith(".gz"):
            _decompress_gz(dest)

    print(f"\n   📂 Files saved to: {EICU_DIR}")


def main():
    print("\n" + "=" * 60)
    print("  Real Clinical Data Downloader")
    print("  (No credentials required — open-access PhysioNet)")
    print("=" * 60)

    download_mimic_demo()
    download_eicu_demo()

    print("\n" + "=" * 60)
    print("  ✅ Downloads complete!")
    print("  Next step:")
    print("     python src/data/build_real_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
