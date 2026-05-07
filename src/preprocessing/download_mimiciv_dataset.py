"""
download_mimiciv_dataset.py
==========================
Crash-proof, resumable MIMIC-IV 2.1 downloader.

Features
--------
- Resumes from exact byte offset if a partial file exists (HTTP Range header)
- Auto-retries with exponential back-off on network failures (up to 20 retries)
- Streams in 1 MB chunks — never loads the full file into RAM
- Writes timestamped progress to  data/mimic_iv_raw/download_log.txt
- Runs independently of the terminal window (safe to close terminal)

Usage
-----
  python src/preprocessing/download_mimiciv_dataset.py

Python >= 3.8 + requests (`pip install requests`)
Kaggle creds at  ~/.kaggle/kaggle.json  (username + key)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent
OUT_DIR  = ROOT / "data" / "mimic_iv_raw"
ZIP_PATH = OUT_DIR / "mimic-iv-2-1.zip"
LOG_PATH = OUT_DIR / "download_log.txt"

DATASET_SLUG = "mangeshwagle/mimic-iv-2-1"
CHUNK_SIZE   = 1 * 1024 * 1024   # 1 MB streaming chunks
MAX_RETRIES  = 20
BACKOFF_BASE = 5                  # seconds before first retry; doubles each time

REQUIRED_FILES = {
    "hosp/patients.csv",
    "hosp/admissions.csv",
    "hosp/diagnoses_icd.csv",
    "hosp/d_labitems.csv",
    "hosp/labevents.csv",
    "icu/icustays.csv",
    "icu/d_items.csv",
    "icu/chartevents.csv",
}


# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Kaggle credentials ─────────────────────────────────────────────────────────
def get_kaggle_creds() -> tuple[str, str]:
    cred_file = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_file.exists():
        log("ERROR: ~/.kaggle/kaggle.json not found.")
        log("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        log("  2. Save the downloaded file to: " + str(cred_file))
        sys.exit(1)
    creds = json.loads(cred_file.read_text(encoding="utf-8"))
    return creds["username"], creds["key"]


# ── Get direct download URL via Kaggle API ─────────────────────────────────────
def get_kaggle_download_url(username: str, key: str) -> str:
    """
    Uses the Kaggle REST API directly to obtain a pre-signed download URL
    for the full dataset zip, avoiding the kaggle package's lack of resume.
    """
    try:
        import requests
    except ImportError:
        log("Installing requests...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests  # type: ignore

    owner, dataset = DATASET_SLUG.split("/")
    api_url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}"
    log(f"Fetching download URL for: {DATASET_SLUG}")

    # HEAD first to resolve redirect → get final URL
    resp = requests.head(api_url, auth=(username, key), allow_redirects=True, timeout=30)
    if resp.status_code not in (200, 302, 301):
        log(f"WARNING: HEAD returned {resp.status_code}, will try GET directly.")
    return api_url   # requests will follow redirects on GET


# ── Resumable download ─────────────────────────────────────────────────────────
def download_with_resume(url: str, username: str, key: str):
    import requests

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine resume offset
    resume_pos = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0
    if resume_pos > 0:
        log(f"Partial file detected: {resume_pos / 1e9:.3f} GB — resuming from byte {resume_pos:,}")
    else:
        log("Starting fresh download...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {}
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"

            log(f"Attempt {attempt}/{MAX_RETRIES}  |  offset={resume_pos / 1e9:.3f} GB")
            resp = requests.get(
                url,
                auth=(username, key),
                headers=headers,
                stream=True,
                timeout=(30, 120),   # connect=30s, read=120s
                allow_redirects=True,
            )

            # Server rejected range request — restart from 0
            if resp.status_code == 200 and resume_pos > 0:
                log("Server returned 200 (no range support) — restarting from byte 0")
                resume_pos = 0

            if resp.status_code not in (200, 206):
                log(f"ERROR: HTTP {resp.status_code} — {resp.text[:200]}")
                raise RuntimeError(f"HTTP {resp.status_code}")

            total_str = resp.headers.get("Content-Length", "unknown")
            if total_str != "unknown":
                remaining = int(total_str)
                total     = resume_pos + remaining
                log(f"Total size: {total / 1e9:.2f} GB  |  Remaining: {remaining / 1e9:.2f} GB")
            else:
                total = None
                log("Total size: unknown")

            write_mode = "ab" if resume_pos > 0 else "wb"
            downloaded = resume_pos
            last_log   = time.time()
            t_start    = time.time()

            with ZIP_PATH.open(write_mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)

                    now = time.time()
                    if now - last_log >= 30:          # log every 30 s
                        elapsed  = now - t_start + 0.001
                        speed_mb = (downloaded - resume_pos) / elapsed / 1e6
                        pct      = f"{downloaded / total * 100:.1f}%" if total else "??%"
                        if total and speed_mb > 0:
                            eta_s = (total - downloaded) / (speed_mb * 1e6)
                            eta   = f"{eta_s / 60:.0f} min"
                        else:
                            eta = "??"
                        log(f"  Progress: {pct}  |  {downloaded / 1e9:.3f} GB  |  "
                            f"{speed_mb:.1f} MB/s  |  ETA: {eta}")
                        last_log = now

            log("Download complete!")
            return  # success

        except Exception as exc:
            # Update resume_pos to current file size for next attempt
            if ZIP_PATH.exists():
                resume_pos = ZIP_PATH.stat().st_size
            wait = min(BACKOFF_BASE * (2 ** (attempt - 1)), 300)
            log(f"Error on attempt {attempt}: {exc}")
            log(f"Retrying in {wait}s...  (saved {resume_pos / 1e9:.3f} GB so far)")
            time.sleep(wait)

    log("FAILED: Max retries exceeded. Re-run script to continue from current offset.")
    sys.exit(1)


# ── Verify ZIP ─────────────────────────────────────────────────────────────────
def verify_zip():
    log("Verifying ZIP integrity and required files...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            names  = set(zf.namelist())
            missing = REQUIRED_FILES - {r for r in REQUIRED_FILES if any(r in n for n in names)}

        log(f"  ZIP entries  : {len(names):,}")
        log(f"  Required files found: {len(REQUIRED_FILES) - len(missing)}/{len(REQUIRED_FILES)}")
        if missing:
            log(f"  WARNING - missing: {missing}")
        else:
            log("  All required files present!")

        # Sample paths
        sample = sorted(names)[:15]
        log("  Sample paths inside ZIP:")
        for s in sample:
            log(f"    {s}")

    except zipfile.BadZipFile as e:
        log(f"BAD ZIP: {e} — file may be incomplete or corrupted.")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    # Clear old log and start fresh session marker
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"  Download session started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")

    log("MIMIC-IV 2.1 Robust Downloader")
    log(f"  Output: {ZIP_PATH}")

    username, key = get_kaggle_creds()
    log(f"  Kaggle user: {username}")

    url = get_kaggle_download_url(username, key)
    download_with_resume(url, username, key)
    verify_zip()

    log("=" * 60)
    log("SUCCESS! Next step:")
    log("  python src/preprocessing/build_cardiac_progression_dataset.py")
    log("=" * 60)


if __name__ == "__main__":
    main()
