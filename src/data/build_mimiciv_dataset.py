"""
build_mimiciv_dataset.py
========================
Extracts cardiac features from the MIMIC-IV 2.1 dataset (67 GB ZIP) and
produces .npy arrays ready for model training — without ever writing the full
67 GB to disk.

Prediction target
-----------------
"Cardiac progression" — for each patient, the model predicts whether their
cardiac condition will WORSEN (label=1) or IMPROVE / STAY STABLE (label=0)
by the next ICU admission.

This is determined by a Cardiac Severity Index (CSI) computed from vitals
and lab values at each ICU stay.  For single-visit patients (or the final
visit), hospital in-hospital mortality is used as the label.

Feature schema (58 features — identical to run_local_pipeline.py)
------------------------------------------------------------------
The same 58-feature vector is produced so that all three existing model files
(train_lstm.py, train_gru.py, train_transformer.py) work without any changes.

Pipeline
--------
Step 1 — Load small tables into RAM (patients, admissions, icustays,
          diagnoses_icd) and build a cardiac patient cohort.
Step 2 — Stream chartevents.csv in chunks (100 k rows at a time) from inside
          the ZIP; keep only cardiac ICU stays × target vital itemids.
Step 3 — Stream labevents.csv in chunks; keep only cardiac admissions ×
          target lab itemids.
Step 4 — Aggregate per ICU stay: mean / std / max per feature.
Step 5 — Merge vitals, labs, and clinical metadata.
Step 6 — Compute Cardiac Severity Index (CSI) per stay.
Step 7 — Build progression labels across successive stays per patient.
Step 8 — Forward-fill missing values, then StandardScaler.
Step 9 — Sliding-window sequences → save .npy to data/preprocessed/.

Usage
-----
python src/data/build_mimiciv_dataset.py [--zip data/mimic_iv_raw/mimic-iv-2-1.zip]
                                         [--chunk 100000]
                                         [--seq_len 6]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent.parent
ZIP_PATH = ROOT / "data" / "mimic_iv_raw" / "mimic-iv-2-1.zip"
REAL_DIR = ROOT / "data" / "real"
PREP_DIR = ROOT / "data" / "preprocessed"


# ══════════════════════════════════════════════════════════════════════════════
# Feature schema — MUST MATCH run_local_pipeline.py exactly (58 features)
# ══════════════════════════════════════════════════════════════════════════════

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
    # Temporal
    "time_delta_days", "visit_index",
]
NUM_FEATURES = len(FEATURE_NAMES)   # 58


# ══════════════════════════════════════════════════════════════════════════════
# Item ID Dictionaries  (verified against MIMIC-IV community references)
# These are augmented at runtime from d_items / d_labitems inside the ZIP.
# ══════════════════════════════════════════════════════════════════════════════

# chartevents (icu/d_items)
VITAL_ITEM_IDS: dict[str, list[int]] = {
    "heart_rate":   [220045],
    "sbp":          [220179, 220050],
    "dbp":          [220180, 220051],
    "mbp":          [220181, 220052],
    "resp_rate":    [220210, 224690],
    "temperature":  [223761, 223762],          # F and C — converted below
    "spo2":         [220277],
    "glucose_fs":   [225664, 220621, 226537],
}

# labevents (hosp/d_labitems)
LAB_ITEM_IDS: dict[str, list[int]] = {
    "troponin_t":   [51003],
    "troponin_i":   [52100, 51002],
    "bnp":          [50816],
    "nt_probnp":    [50963],
    "creatinine":   [50912],
    "bun":          [51006],
    "potassium":    [50971],
    "sodium":       [50983],
    "chloride":     [50902],
    "bicarbonate":  [50882],
    "hematocrit":   [51221],
    "hemoglobin":   [51222],
    "wbc":          [51301],
    "platelets":    [51265],
    "glucose_lab":  [50931],
}

# ICD-9 prefixes (same as original MIMIC-III code)
CARDIAC_ICD9_PREFIXES = [
    "410", "411", "412", "413", "414",  # MI / IHD
    "428",                               # Heart failure
    "426", "427",                        # Arrhythmia
    "425",                               # Cardiomyopathy
    "394", "395", "396", "397", "424",  # Valvular
    "402", "404",                        # Hypertensive HD
    "415", "416", "417",                 # Pulmonary HD
    "429",                               # Other HD
]

# ICD-10 codes (MIMIC-IV has both — support both)
CARDIAC_ICD10_PREFIXES = [
    "I21", "I22", "I23", "I24", "I25",  # MI / IHD
    "I50",                               # Heart failure
    "I44", "I45", "I46", "I47", "I48", "I49",  # Arrhythmia
    "I42",                               # Cardiomyopathy
    "I34", "I35", "I36", "I37",         # Valvular
    "I11", "I13",                        # Hypertensive HD
    "I26", "I27",                        # Pulmonary HD
    "I51",                               # Other HD
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_in_zip(zf: zipfile.ZipFile, pattern: str) -> str | None:
    """Fuzzy-match a filename inside the ZIP."""
    names = zf.namelist()
    # 1. Exact suffix
    m = next((n for n in names if n.endswith(pattern)), None)
    if m:
        return m
    # 2. Basename only
    base = pattern.split("/")[-1]
    m = next((n for n in names if n.endswith(base)), None)
    return m


def _open_df(zf: zipfile.ZipFile, pattern: str,
             usecols=None, parse_dates=None, dtype=None) -> pd.DataFrame:
    """Read an entire small CSV from inside the ZIP."""
    path = _find_in_zip(zf, pattern)
    if path is None:
        raise FileNotFoundError(f"'{pattern}' not found in ZIP. Run explore_mimiciv.py to debug.")
    print(f"   Reading {path!r} …", end=" ", flush=True)
    with zf.open(path) as raw:
        df = pd.read_csv(
            io.TextIOWrapper(raw, encoding="utf-8"),
            usecols=usecols, parse_dates=parse_dates, dtype=dtype,
            low_memory=False,
        )
    df.columns = [c.lower() for c in df.columns]
    print(f"{len(df):,} rows")
    return df


def _is_cardiac(code: str, version: int) -> bool:
    if not isinstance(code, str):
        return False
    code = code.strip().upper()
    if version == 9:
        return any(code.startswith(p) for p in CARDIAC_ICD9_PREFIXES)
    else:  # version 10
        return any(code.startswith(p) for p in CARDIAC_ICD10_PREFIXES)


def _celsius_to_fahrenheit(val: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""
    return val * 9 / 5 + 32


# ══════════════════════════════════════════════════════════════════════════════
# Cardiac Severity Index (CSI)
# ══════════════════════════════════════════════════════════════════════════════

def compute_csi(row: pd.Series) -> float:
    """
    Composite Cardiac Severity Index.
    Each component contributes 0–3 points; higher = more severe.
    Designed to be monotonically increasing with cardiac deterioration.

    Components:
        1. Heart rate deviation from normal (60–100 bpm)
        2. Systolic BP: hypotension < 90 mmHg (emergency) or hypertension > 160
        3. Troponin T (myocardial injury marker)
        4. BNP / NT-proBNP (HF severity)
        5. SpO2 (oxygenation)
        6. Creatinine (cardiorenal syndrome)
        7. Respiratory rate (tachypnea = decompensation)
        8. Hemoglobin (anemia worsens cardiac load)
    """
    score = 0.0

    # 1. Heart rate
    hr = row.get("heart_rate_mean", np.nan)
    if pd.notna(hr):
        if hr < 60:
            score += min(3.0, (60 - hr) / 20)      # bradycardia
        elif hr > 100:
            score += min(3.0, (hr - 100) / 40)     # tachycardia

    # 2. Systolic BP
    sbp = row.get("sbp_mean", np.nan)
    if pd.notna(sbp):
        if sbp < 90:
            score += min(3.0, (90 - sbp) / 30)     # hypotension — critical
        elif sbp > 160:
            score += min(1.5, (sbp - 160) / 60)    # hypertension

    # 3. Troponin T (normal < 0.01 ng/mL)
    trop_t = row.get("troponin_t_mean", np.nan)
    if pd.notna(trop_t) and trop_t > 0:
        score += min(3.0, trop_t / 0.04)

    # 4. BNP (normal < 100 pg/mL)
    bnp = row.get("bnp_mean", np.nan)
    if pd.notna(bnp) and bnp > 0:
        score += min(3.0, max(0.0, bnp - 100) / 600)

    # 4b. NT-proBNP (normal < 300 pg/mL)
    nt_probnp = row.get("nt_probnp_mean", np.nan)
    if pd.notna(nt_probnp) and nt_probnp > 0 and pd.isna(bnp):
        score += min(3.0, max(0.0, nt_probnp - 300) / 1800)

    # 5. SpO2 (normal > 95%)
    spo2 = row.get("spo2_mean", np.nan)
    if pd.notna(spo2):
        score += min(3.0, max(0.0, 95 - spo2) / 8)

    # 6. Creatinine (normal ~1.0 mg/dL; >1.5 indicates cardiorenal)
    creat = row.get("creatinine_mean", np.nan)
    if pd.notna(creat):
        score += min(2.0, max(0.0, creat - 1.5) / 2.0)

    # 7. Respiratory rate (normal 12–20 breaths/min)
    rr = row.get("resp_rate_mean", np.nan)
    if pd.notna(rr):
        score += min(2.0, max(0.0, rr - 20) / 10)

    # 8. Hemoglobin (normal ~12–16 g/dL; low → increased cardiac load)
    hgb = row.get("hemoglobin_mean", np.nan)
    if pd.notna(hgb):
        score += min(2.0, max(0.0, 12 - hgb) / 6)

    return round(score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Small tables
# ══════════════════════════════════════════════════════════════════════════════

def load_cohort(zf: zipfile.ZipFile) -> tuple[set, set, set, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        cardiac_subject_ids  — set of subject_id
        cardiac_hadm_ids     — set of hadm_id
        cardiac_stay_ids     — set of stay_id
        icu_df               — icustays with merged patient/admission metadata
        diag_counts          — DataFrame[hadm_id, num_diagnoses]
    """
    print("\n── Step 1: Loading small tables ──────────────────────────────")

    # patients
    patients = _open_df(zf, "hosp/patients.csv",
                        usecols=["subject_id", "gender", "anchor_age",
                                 "anchor_year", "dod"])

    # admissions
    admissions = _open_df(zf, "hosp/admissions.csv",
                          usecols=["subject_id", "hadm_id", "admittime",
                                   "dischtime", "admission_type",
                                   "hospital_expire_flag"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    # diagnoses_icd
    diag = _open_df(zf, "hosp/diagnoses_icd.csv",
                    usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])

    # icustays
    icustays = _open_df(zf, "icu/icustays.csv",
                        usecols=["subject_id", "hadm_id", "stay_id",
                                 "intime", "outtime", "los"])
    icustays["intime"]  = pd.to_datetime(icustays["intime"],  errors="coerce")
    icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")

    # ── Filter cardiac cohort (ICD-9 + ICD-10) ──────────────────────────────
    diag["is_cardiac"] = diag.apply(
        lambda r: _is_cardiac(r["icd_code"], r.get("icd_version", 9)), axis=1
    )
    cardiac_hadms    = set(diag[diag["is_cardiac"]]["hadm_id"])
    cardiac_subjects = set(diag[diag["is_cardiac"]]["subject_id"])

    # Prior condition flags
    def _has_code(sub_id: int, prefix: str, version: int) -> bool:
        sub = diag[diag["subject_id"] == sub_id]
        codes = sub[sub["icd_version"] == version]["icd_code"].str.upper().str.strip()
        return codes.str.startswith(prefix).any()

    prior_mi_icd9   = set(diag[(diag["icd_version"] == 9) & (diag["icd_code"].str.startswith("412", na=False))]["subject_id"])
    prior_mi_icd10  = set(diag[(diag["icd_version"] == 10) & (diag["icd_code"].str.startswith("I25.2", na=False))]["subject_id"])
    prior_hf_icd9   = set(diag[(diag["icd_version"] == 9) & (diag["icd_code"].str.startswith("428", na=False))]["subject_id"])
    prior_hf_icd10  = set(diag[(diag["icd_version"] == 10) & (diag["icd_code"].str.startswith("I50", na=False))]["subject_id"])
    prior_arr_icd9  = set(diag[(diag["icd_version"] == 9) & (diag["icd_code"].str.startswith(("426", "427"), na=False))]["subject_id"])
    prior_arr_icd10 = set(diag[(diag["icd_version"] == 10) & (diag["icd_code"].str.startswith(("I44", "I48"), na=False))]["subject_id"])

    prior_mi  = prior_mi_icd9  | prior_mi_icd10
    prior_hf  = prior_hf_icd9  | prior_hf_icd10
    prior_arr = prior_arr_icd9 | prior_arr_icd10

    # Diagnosis counts per admission
    diag_count = diag.groupby("hadm_id")["icd_code"].count().rename("num_diagnoses").reset_index()

    # ── Filter ICU stays to cardiac patients ──────────────────────────────────
    icu_cardiac = icustays[
        icustays["subject_id"].isin(cardiac_subjects) |
        icustays["hadm_id"].isin(cardiac_hadms)
    ].copy()

    cardiac_stay_ids = set(icu_cardiac["stay_id"])

    print(f"\n   Cardiac subjects   : {len(cardiac_subjects):,}")
    print(f"   Cardiac admissions : {len(cardiac_hadms):,}")
    print(f"   Cardiac ICU stays  : {len(cardiac_stay_ids):,}")

    # ── Merge metadata ────────────────────────────────────────────────────────
    adm_cols = ["hadm_id", "admittime", "dischtime",
                "admission_type", "hospital_expire_flag"]
    icu = icu_cardiac.merge(admissions[adm_cols], on="hadm_id", how="left")
    icu = icu.merge(patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
                    on="subject_id", how="left")
    icu = icu.merge(diag_count, on="hadm_id", how="left")

    # Age: MIMIC-IV uses anchor_age (age at anchor_year) not exact DOB
    icu["age"] = icu["anchor_age"].clip(18, 120)

    # LOS in days (icustays.los is already in days for MIMIC-IV)
    icu["los_days"] = icu["los"].clip(0, 365)

    # Gender
    icu["gender_male"] = (icu["gender"].str.upper() == "M").astype(int)

    # Admission type flags
    icu["admission_type_emergency"] = icu["admission_type"].str.upper().str.contains(
        "EMERGENCY|URGENT", na=False).astype(int)
    icu["admission_type_elective"]  = icu["admission_type"].str.upper().str.contains(
        "ELECTIVE", na=False).astype(int)

    # Prior conditions
    icu["prior_mi"]         = icu["subject_id"].isin(prior_mi).astype(int)
    icu["prior_hf"]         = icu["subject_id"].isin(prior_hf).astype(int)
    icu["prior_arrhythmia"] = icu["subject_id"].isin(prior_arr).astype(int)
    icu["icu_flag"]         = 1
    icu["num_diagnoses"]    = icu["num_diagnoses"].fillna(1)

    # Days since last admission (per patient, chronological)
    icu = icu.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    icu["days_since_last_admission"] = (
        icu.groupby("subject_id")["intime"]
        .diff()
        .dt.total_seconds()
        .div(86400)
        .clip(0)
        .fillna(0)
    )

    # Time delta in days from first ICU stay (per patient)
    first_times = icu.groupby("subject_id")["intime"].transform("min")
    icu["time_delta_days"] = (
        (icu["intime"] - first_times).dt.total_seconds() / 86400
    ).clip(0).fillna(0)

    # Visit index (0-indexed chronological counter per patient)
    icu["visit_index"] = icu.groupby("subject_id").cumcount()

    # Hospital expire flag (label seed for single-visit patients)
    icu["hospital_expire_flag"] = icu["hospital_expire_flag"].fillna(0).astype(int)

    return cardiac_subject_ids, cardiac_hadms, cardiac_stay_ids, icu, diag_count


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Chunked vitals extraction from chartevents
# ══════════════════════════════════════════════════════════════════════════════

def extract_vitals(zf: zipfile.ZipFile, cardiac_stay_ids: set,
                   chunk_size: int = 100_000) -> pd.DataFrame:
    """
    Stream chartevents.csv in chunks.
    Keep only rows for cardiac ICU stays and target itemids.
    Aggregate: mean, std per (stay_id, vital).
    """
    print("\n── Step 2: Extracting vitals from chartevents (chunked) ──────")

    all_item_ids = {iid for ids in VITAL_ITEM_IDS.values() for iid in ids}
    item_to_feat: dict[int, str] = {}
    for feat, ids in VITAL_ITEM_IDS.items():
        for iid in ids:
            item_to_feat[iid] = feat

    path = _find_in_zip(zf, "icu/chartevents.csv")
    if path is None:
        print("   ⚠️  chartevents.csv not found — vitals will be NaN")
        return pd.DataFrame(columns=["stay_id"])

    # Determine compressed size for progress estimation
    info      = zf.getinfo(path)
    comp_mb   = info.compress_size / 1e6
    uncomp_mb = info.file_size / 1e6
    print(f"   chartevents: {comp_mb:.0f} MB compressed / {uncomp_mb:.0f} MB uncompressed")
    print(f"   Chunk size : {chunk_size:,} rows")

    keep_cols = ["stay_id", "itemid", "valuenum"]
    rows_read  = 0
    rows_kept  = 0
    agg_parts: list[pd.DataFrame] = []

    with zf.open(path) as raw:
        reader = pd.read_csv(
            io.TextIOWrapper(raw, encoding="utf-8"),
            usecols=keep_cols,
            dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": float},
            chunksize=chunk_size,
            low_memory=False,
        )
        for chunk in tqdm(reader, desc="   chartevents chunks", unit="chunk"):
            rows_read += len(chunk)
            # Filter
            chunk = chunk[
                chunk["stay_id"].isin(cardiac_stay_ids) &
                chunk["itemid"].isin(all_item_ids)
            ]
            if chunk.empty:
                continue
            chunk["feat"] = chunk["itemid"].map(item_to_feat)
            rows_kept += len(chunk)

            # Convert Celsius temperature to Fahrenheit
            temp_c_ids = set(VITAL_ITEM_IDS.get("temperature", []))
            # Heuristic: values < 50 are likely Celsius
            is_temp = chunk["feat"] == "temperature"
            is_cold = chunk["valuenum"] < 50
            chunk.loc[is_temp & is_cold, "valuenum"] = (
                chunk.loc[is_temp & is_cold, "valuenum"] * 9 / 5 + 32
            )

            agg_parts.append(chunk[["stay_id", "feat", "valuenum"]])

    print(f"\n   Rows read  : {rows_read:,}")
    print(f"   Rows kept  : {rows_kept:,} ({rows_kept/max(rows_read,1)*100:.1f}%)")

    if not agg_parts:
        print("   ⚠️  No vital rows matched — check item IDs with explore_mimiciv.py")
        return pd.DataFrame(columns=["stay_id"])

    all_vitals = pd.concat(agg_parts, ignore_index=True)
    all_vitals = all_vitals.dropna(subset=["valuenum"])

    # Aggregate per stay × feature
    grp = all_vitals.groupby(["stay_id", "feat"])["valuenum"]
    vitals_agg = pd.DataFrame({
        "mean": grp.mean(),
        "std":  grp.std().fillna(0),
        "max":  grp.max(),
    }).reset_index()

    # Pivot so that each row = one stay, columns = feat_mean / feat_std / feat_max
    pivot_mean = vitals_agg.pivot(index="stay_id", columns="feat", values="mean")
    pivot_std  = vitals_agg.pivot(index="stay_id", columns="feat", values="std")
    pivot_max  = vitals_agg.pivot(index="stay_id", columns="feat", values="max")

    pivot_mean.columns = [f"{c}_mean" for c in pivot_mean.columns]
    pivot_std.columns  = [f"{c}_std"  for c in pivot_std.columns]
    pivot_max.columns  = [f"{c}_max"  for c in pivot_max.columns]

    vitals_out = pd.concat([pivot_mean, pivot_std, pivot_max], axis=1).reset_index()

    # Rename to match FEATURE_NAMES (glucose_fs → glucose_fingerstick)
    vitals_out = vitals_out.rename(columns={
        c: c.replace("glucose_fs_", "glucose_fingerstick_")
        for c in vitals_out.columns
    })

    print(f"   Unique ICU stays with vitals: {vitals_out['stay_id'].nunique():,}")
    return vitals_out


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Chunked labs extraction from labevents
# ══════════════════════════════════════════════════════════════════════════════

def extract_labs(zf: zipfile.ZipFile, cardiac_hadm_ids: set,
                 chunk_size: int = 100_000) -> pd.DataFrame:
    """
    Stream labevents.csv in chunks.
    Keep only rows for cardiac admissions and target itemids.
    Aggregate: mean, std, max per (hadm_id, lab).
    """
    print("\n── Step 3: Extracting labs from labevents (chunked) ──────────")

    all_lab_ids = {iid for ids in LAB_ITEM_IDS.values() for iid in ids}
    item_to_lab: dict[int, str] = {}
    for feat, ids in LAB_ITEM_IDS.items():
        for iid in ids:
            item_to_lab[iid] = feat

    path = _find_in_zip(zf, "hosp/labevents.csv")
    if path is None:
        print("   ⚠️  labevents.csv not found — labs will be NaN")
        return pd.DataFrame(columns=["hadm_id"])

    info      = zf.getinfo(path)
    comp_mb   = info.compress_size / 1e6
    uncomp_mb = info.file_size / 1e6
    print(f"   labevents: {comp_mb:.0f} MB compressed / {uncomp_mb:.0f} MB uncompressed")

    keep_cols = ["hadm_id", "itemid", "valuenum"]
    rows_read  = 0
    rows_kept  = 0
    agg_parts: list[pd.DataFrame] = []

    with zf.open(path) as raw:
        reader = pd.read_csv(
            io.TextIOWrapper(raw, encoding="utf-8"),
            usecols=keep_cols,
            dtype={"hadm_id": "Int64", "itemid": "Int64", "valuenum": float},
            chunksize=chunk_size,
            low_memory=False,
        )
        for chunk in tqdm(reader, desc="   labevents chunks  ", unit="chunk"):
            rows_read += len(chunk)
            chunk = chunk.dropna(subset=["hadm_id"])
            chunk = chunk[
                chunk["hadm_id"].isin(cardiac_hadm_ids) &
                chunk["itemid"].isin(all_lab_ids)
            ]
            if chunk.empty:
                continue
            chunk["feat"] = chunk["itemid"].map(item_to_lab)
            rows_kept += len(chunk)
            agg_parts.append(chunk[["hadm_id", "feat", "valuenum"]])

    print(f"\n   Rows read : {rows_read:,}")
    print(f"   Rows kept : {rows_kept:,} ({rows_kept/max(rows_read,1)*100:.1f}%)")

    if not agg_parts:
        print("   ⚠️  No lab rows matched.")
        return pd.DataFrame(columns=["hadm_id"])

    all_labs = pd.concat(agg_parts, ignore_index=True)
    all_labs = all_labs.dropna(subset=["valuenum"])

    # Sanity clip — remove physiologically impossible values
    # (outliers common in EHR data)
    CLIP_LIMITS = {
        "creatinine":  (0.1,  30.0),
        "bun":         (1.0, 200.0),
        "potassium":   (1.5,  10.0),
        "sodium":      (100.0, 180.0),
        "hematocrit":  (5.0,  70.0),
        "hemoglobin":  (2.0,  25.0),
        "wbc":         (0.1, 100.0),
        "platelets":   (5.0, 2000.0),
        "glucose_lab": (20.0, 2000.0),
        "troponin_t":  (0.0, 100.0),
        "troponin_i":  (0.0, 1000.0),
        "bnp":         (0.0, 50000.0),
    }
    for feat, (lo, hi) in CLIP_LIMITS.items():
        mask = all_labs["feat"] == feat
        all_labs.loc[mask, "valuenum"] = all_labs.loc[mask, "valuenum"].clip(lo, hi)

    # Aggregate
    grp = all_labs.groupby(["hadm_id", "feat"])["valuenum"]
    labs_agg = pd.DataFrame({
        "mean": grp.mean(),
        "std":  grp.std().fillna(0),
        "max":  grp.max(),
    }).reset_index()

    pivot_mean = labs_agg.pivot(index="hadm_id", columns="feat", values="mean")
    pivot_std  = labs_agg.pivot(index="hadm_id", columns="feat", values="std")
    pivot_max  = labs_agg.pivot(index="hadm_id", columns="feat", values="max")

    pivot_mean.columns = [f"{c}_mean" for c in pivot_mean.columns]
    pivot_std.columns  = [f"{c}_std"  for c in pivot_std.columns]
    pivot_max.columns  = [f"{c}_max"  for c in pivot_max.columns]

    labs_out = pd.concat([pivot_mean, pivot_std, pivot_max], axis=1).reset_index()
    print(f"   Unique admissions with labs: {labs_out['hadm_id'].nunique():,}")
    return labs_out


# ══════════════════════════════════════════════════════════════════════════════
# Steps 4-7 — Merge, CSI, Progression Labels
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_label(icu_df: pd.DataFrame,
                    vitals_df: pd.DataFrame,
                    labs_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all features and compute progression labels."""

    print("\n── Step 4-7: Merging, CSI computation, labelling ─────────────")

    # Merge vitals (per stay_id) and labs (per hadm_id)
    df = icu_df.copy()
    if not vitals_df.empty and "stay_id" in vitals_df.columns:
        df = df.merge(vitals_df, on="stay_id", how="left")
    if not labs_df.empty and "hadm_id" in labs_df.columns:
        df = df.merge(labs_df, on="hadm_id", how="left")

    # Deduplicate columns from merge
    df = df.loc[:, ~df.columns.duplicated()]

    # ── num_procedures: placeholder (use diag count proxy) ────────────────────
    df["num_procedures"] = df.get("num_diagnoses", pd.Series(1, index=df.index)).clip(0, 50)

    # ── num_medications: not in MIMIC-IV without prescriptions table ──────────
    # Use a simulated value proportional to num_diagnoses
    df["num_medications"] = (df["num_procedures"] * 1.5).clip(1, 30).round()

    # ── Compute CSI ───────────────────────────────────────────────────────────
    print("   Computing Cardiac Severity Index (CSI) per stay …")
    df["csi"] = df.apply(compute_csi, axis=1)
    print(f"   CSI range: {df['csi'].min():.2f} – {df['csi'].max():.2f}  "
          f"(mean={df['csi'].mean():.2f})")

    # ── Progression labels ────────────────────────────────────────────────────
    # Sort by patient + ICU admission time
    df = df.sort_values(["subject_id", "intime"]).reset_index(drop=True)

    labels = np.zeros(len(df), dtype=np.int64)
    df_idx = df.index.tolist()

    # Group by patient
    for _, grp in df.groupby("subject_id"):
        idx = grp.index.tolist()
        for pos, ix in enumerate(idx):
            if pos < len(idx) - 1:
                # Next visit exists: progressive worsening if CSI increases >10%
                curr_csi = df.loc[ix, "csi"]
                next_csi = df.loc[idx[pos + 1], "csi"]
                # Label 1 = worsening, 0 = stable/improving
                labels[ix] = int(next_csi > curr_csi * 1.10)
            else:
                # Last (or only) visit: use hospital mortality
                labels[ix] = int(df.loc[ix, "hospital_expire_flag"])

    df["label"] = labels
    pos_rate = labels.mean()
    print(f"   Label distribution: {labels.sum():,} worsening ({pos_rate:.1%}), "
          f"{(1 - labels).sum():,} stable/improving ({1-pos_rate:.1%})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 8-9 — Build sequences, scale, save
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save(df: pd.DataFrame, seq_len: int = 6):
    """
    Forward-fill → impute → scale → sliding-window sequences → save .npy.
    seq_len: number of consecutive ICU stays to use as one sequence.
    """
    print(f"\n── Step 8-9: Sequences (seq_len={seq_len}), scaling, saving ──")

    # ── Ensure all 58 feature columns exist ───────────────────────────────────
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = np.nan

    df = df.sort_values(["subject_id", "intime"]).reset_index(drop=True)

    # ── Forward-fill within patient, then global median imputation ────────────
    print("   Imputing missing values …")
    for feat in FEATURE_NAMES:
        df[feat] = (
            df.groupby("subject_id")[feat]
            .transform(lambda x: x.ffill().bfill())
        )
    medians = df[FEATURE_NAMES].median()
    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(medians)

    # ── Save fused CSV ────────────────────────────────────────────────────────
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = REAL_DIR / "mimiciv_longitudinal_features.csv"
    df[["subject_id", "stay_id", "hadm_id", "visit_index", "label", "csi"] +
       FEATURE_NAMES].to_csv(csv_out, index=False)
    print(f"   💾 Saved fused CSV → {csv_out}")

    # ── Sliding-window sequences ──────────────────────────────────────────────
    X_list: list[np.ndarray] = []
    y_list: list[int]        = []

    for pid, grp in df.groupby("subject_id"):
        grp = grp.sort_values("visit_index").reset_index(drop=True)
        vals   = grp[FEATURE_NAMES].values.astype(np.float32)
        labels_ = grp["label"].values

        n = len(grp)
        if n < seq_len:
            # Pad with first row if not enough history
            pad = seq_len - n
            vals    = np.vstack([np.tile(vals[[0]], (pad, 1)), vals])
            labels_ = np.concatenate([[labels_[0]] * pad, labels_])
            n = seq_len

        # One sample per position
        for i in range(n - seq_len + 1):
            X_list.append(vals[i: i + seq_len])
            y_list.append(int(labels_[i + seq_len - 1]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"\n   Sequences: {len(y):,}  shape: {X.shape}")

    if len(y) < 50:
        print("❌  Too few sequences. Check cardiac cohort detection.")
        sys.exit(1)

    # ── Train / Val / Test split ──────────────────────────────────────────────
    try:
        X_tv,  X_test,  y_tv,  y_test  = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.15 / 0.85, random_state=42, stratify=y_tv)
    except ValueError:
        X_tv,  X_test,  y_tv,  y_test  = train_test_split(
            X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.15 / 0.85, random_state=42)

    # ── Scale (fit on train only) ─────────────────────────────────────────────
    n_tr, T, F = X_train.shape
    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(X_train.reshape(-1, F)).reshape(n_tr, T, F)
    X_val       = scaler.transform(X_val.reshape(-1,   F)).reshape(X_val.shape)
    X_test      = scaler.transform(X_test.reshape(-1,  F)).reshape(X_test.shape)

    # ── Save ──────────────────────────────────────────────────────────────────
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREP_DIR / "X_train.npy", X_train)
    np.save(PREP_DIR / "X_val.npy",   X_val)
    np.save(PREP_DIR / "X_test.npy",  X_test)
    np.save(PREP_DIR / "y_train.npy", y_train)
    np.save(PREP_DIR / "y_val.npy",   y_val)
    np.save(PREP_DIR / "y_test.npy",  y_test)
    joblib.dump(scaler,       PREP_DIR / "preprocessor.pkl")
    joblib.dump(FEATURE_NAMES, PREP_DIR / "feature_names.pkl")

    print(f"\n📊  Final Dataset Shapes:")
    print(f"   X_train : {X_train.shape}   y_train : {y_train.shape}  "
          f"pos={y_train.sum()} ({y_train.mean():.1%})")
    print(f"   X_val   : {X_val.shape}     y_val   : {y_val.shape}    "
          f"pos={y_val.sum()} ({y_val.mean():.1%})")
    print(f"   X_test  : {X_test.shape}    y_test  : {y_test.shape}   "
          f"pos={y_test.sum()} ({y_test.mean():.1%})")
    print(f"\n💾  Saved .npy arrays to: {PREP_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-IV cardiac features and build training arrays"
    )
    parser.add_argument("--zip", type=Path, default=ZIP_PATH,
                        help=f"Path to MIMIC-IV 2.1 ZIP (default: {ZIP_PATH})")
    parser.add_argument("--chunk", type=int, default=100_000,
                        help="Rows per CSV chunk (default: 100,000 — reduce if low RAM)")
    parser.add_argument("--seq_len", type=int, default=6,
                        help="Sequence length (consecutive ICU stays, default: 6)")
    args = parser.parse_args()

    if not args.zip.exists():
        print(f"\n❌  ZIP not found: {args.zip}")
        print("    Run: python src/data/download_mimiciv.py")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  MIMIC-IV 2.1 — Cardiac Progression Dataset Builder")
    print(f"  ZIP      : {args.zip}")
    print(f"  Chunk    : {args.chunk:,} rows")
    print(f"  Seq len  : {args.seq_len} ICU stays per sequence")
    print(f"  Target   : Cardiac progression (worsening vs stable/improving)")
    print(f"  Features : {NUM_FEATURES}")
    print("=" * 65)

    t_total = time.time()

    with zipfile.ZipFile(args.zip, "r") as zf:

        # ── Steps 1 ────────────────────────────────────────────────────────────
        cardiac_subject_ids, cardiac_hadm_ids, cardiac_stay_ids, icu_df, _ = \
            load_cohort(zf)

        # ── Steps 2–3 ─────────────────────────────────────────────────────────
        vitals_df = extract_vitals(zf, cardiac_stay_ids, chunk_size=args.chunk)
        labs_df   = extract_labs(zf,   cardiac_hadm_ids,  chunk_size=args.chunk)

    # ── Steps 4–9 (no more zip needed) ────────────────────────────────────────
    df = merge_and_label(icu_df, vitals_df, labs_df)
    build_and_save(df, seq_len=args.seq_len)

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*65}")
    print(f"  ✅  Dataset build complete in {elapsed:.1f} min")
    print(f"  Next steps:")
    print(f"    python src/models/train_lstm.py")
    print(f"    python src/models/train_gru.py")
    print(f"    python src/models/train_transformer.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
