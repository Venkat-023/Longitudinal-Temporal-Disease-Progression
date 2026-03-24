"""
build_real_dataset.py
======================
Extracts temporal cardiac features from MIMIC-III Demo + eICU Demo,
fuses them into a shared schema, and saves preprocessed .npy arrays
ready for model training.

Pipeline
--------
1. MIMIC-III arm  : cardiac ICD-9 filter → vitals + labs per ICU stay
2. eICU arm       : cardiac diagnosis filter → periodic vitals + labs per unit stay
3. Fusion         : align to 30 shared features, combine, sort by patient+visit
4. Preprocess     : forward-fill → StandardScaler → sliding windows → split

Usage
-----
python src/data/build_real_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT       = Path(__file__).resolve().parent.parent.parent
MIMIC_DIR  = ROOT / "data" / "mimic_demo"
EICU_DIR   = ROOT / "data" / "eicu_demo"
REAL_DIR   = ROOT / "data" / "real"
PREP_DIR   = ROOT / "data" / "preprocessed"

# ─────────────────────────────────────────────────────────────────────────────
# Shared feature schema (30 features)
# Both arms will produce a DataFrame with exactly these columns
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Vitals
    "heart_rate",
    "sbp",
    "dbp",
    "resp_rate",
    "temperature",
    "spo2",
    "glucose",
    # Labs
    "creatinine",
    "bun",
    "potassium",
    "sodium",
    "hematocrit",
    "hemoglobin",
    "wbc",
    "platelets",
    "bicarbonate",
    # Cardiac markers (often NaN in demo — OK, imputed)
    "troponin",
    "bnp",
    # Clinical context
    "age",
    "los_hours",
    "icu_flag",
    "gender_male",
    "num_diagnoses",
    "apache_score",
    # Temporal
    "visit_index",
    "days_since_admission",
    # Prior conditions
    "prior_mi",
    "prior_hf",
    "prior_arrhythmia",
    # Outcome-adjacent (severity)
    "hospital_expire_flag",
]
NUM_FEATURES = len(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# ICD-9 cardiac code prefixes
# ─────────────────────────────────────────────────────────────────────────────
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

# eICU cardiac keyword patterns
CARDIAC_KEYWORDS = [
    "heart failure", "myocardial", "arrhythmia",
    "cardiac arrest", "atrial fibrillation", "coronary",
    "angina", "cardiomyopathy", "ventricular",
    "bradycardia", "tachycardia", "aortic stenosis",
    "mitral", "endocarditis", "pericarditis",
]


def _is_cardiac_icd9(code: str) -> bool:
    if not isinstance(code, str):
        return False
    code = code.strip().upper()
    return any(code.startswith(p) for p in CARDIAC_ICD9_PREFIXES)


def _safe_mean(series: pd.Series) -> float:
    return series.dropna().mean() if not series.dropna().empty else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# MIMIC-III DEMO ARM
# ═══════════════════════════════════════════════════════════════════════════════

# CHARTEVENTS item IDs → feature name
VITAL_ITEMS = {
    "heart_rate":   [211, 220045],
    "sbp":         [51, 442, 455, 220179, 220050],
    "dbp":         [8368, 8440, 220180, 220051],
    "resp_rate":   [618, 615, 220210],
    "temperature": [223761, 678, 223762, 676],
    "spo2":        [646, 220277],
    "glucose":     [807, 811, 1529, 3745, 225664, 220621],
}

LAB_ITEMS = {
    "creatinine":  [50912],
    "bun":         [51006],
    "potassium":   [50971],
    "sodium":      [50983],
    "hematocrit":  [51221],
    "hemoglobin":  [51222],
    "wbc":         [51301],
    "platelets":   [51265],
    "bicarbonate": [50882],
    "troponin":    [51002, 227429],
    "bnp":         [51006],
    "glucose_lab": [50931],
}


def build_mimic_arm() -> pd.DataFrame:
    """Extract temporal cardiac features from MIMIC-III Demo."""
    print("\n── MIMIC-III Demo Arm ──────────────────────────────────────")

    for f in ["PATIENTS.csv", "ADMISSIONS.csv", "DIAGNOSES_ICD.csv",
              "ICUSTAYS.csv", "CHARTEVENTS.csv", "LABEVENTS.csv"]:
        if not (MIMIC_DIR / f).exists():
            print(f"   ⚠️  Missing {f} — skipping MIMIC arm.")
            print(f"       Run: python src/data/download_real_data.py")
            return pd.DataFrame()

    # Load all columns, then uppercase (demo CSVs use lowercase names)
    def _load_upper(path, parse_dates=None):
        df = pd.read_csv(MIMIC_DIR / path, low_memory=False)
        df.columns = [c.upper() for c in df.columns]
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    patients   = _load_upper("PATIENTS.csv", parse_dates=["DOB", "DOD"])
    admissions = _load_upper("ADMISSIONS.csv", parse_dates=["ADMITTIME", "DISCHTIME"])
    diagnoses  = _load_upper("DIAGNOSES_ICD.csv")
    icustays   = _load_upper("ICUSTAYS.csv", parse_dates=["INTIME", "OUTTIME"])
    chartevents = _load_upper("CHARTEVENTS.csv")
    labevents  = _load_upper("LABEVENTS.csv")

    print(f"   Loaded MIMIC: {len(patients)} patients, "
          f"{len(admissions)} admissions, {len(icustays)} ICU stays")

    # ── Filter cardiac cohort ──────────────────────────────────────
    diagnoses["is_cardiac"] = diagnoses["ICD9_CODE"].apply(_is_cardiac_icd9)
    cardiac_hadms = set(diagnoses[diagnoses["is_cardiac"]]["HADM_ID"])
    cardiac_subjects = set(diagnoses[diagnoses["is_cardiac"]]["SUBJECT_ID"])

    adm_cardiac = admissions[admissions["SUBJECT_ID"].isin(cardiac_subjects)].copy()
    icu_cardiac = icustays[icustays["HADM_ID"].isin(cardiac_hadms)].copy()
    print(f"   Cardiac patients : {admissions[admissions['HADM_ID'].isin(cardiac_hadms)]['SUBJECT_ID'].nunique()}")
    print(f"   Cardiac ICU stays: {len(icu_cardiac)}")

    if icu_cardiac.empty:
        print("   ⚠️  No cardiac ICU stays found — using all ICU stays.")
        icu_cardiac = icustays.copy()

    # ── Prior condition flags (per patient across all admissions) ──
    prior_mi  = set(diagnoses[diagnoses["ICD9_CODE"].str.startswith("412", na=False)]["SUBJECT_ID"])
    prior_hf  = set(diagnoses[diagnoses["ICD9_CODE"].str.startswith("428", na=False)]["SUBJECT_ID"])
    prior_arr = set(diagnoses[diagnoses["ICD9_CODE"].str.startswith(("426", "427"), na=False)]["SUBJECT_ID"])

    # ── Count diagnoses per admission ─────────────────────────────
    diag_count = diagnoses.groupby("HADM_ID")["ICD9_CODE"].count().rename("num_diagnoses")

    # ── Aggregate vitals per ICU stay ─────────────────────────────
    all_item_ids = {i for ids in VITAL_ITEMS.values() for i in ids}
    chart_filt = chartevents[chartevents["ITEMID"].isin(all_item_ids)].copy()
    chart_filt["ICUSTAY_ID"] = pd.to_numeric(chart_filt["ICUSTAY_ID"], errors="coerce")
    chart_filt = chart_filt.dropna(subset=["ICUSTAY_ID"])
    chart_filt["ICUSTAY_ID"] = chart_filt["ICUSTAY_ID"].astype(int)

    vital_agg = {}
    for feat, item_ids in VITAL_ITEMS.items():
        sub = chart_filt[chart_filt["ITEMID"].isin(item_ids)]
        agg = sub.groupby("ICUSTAY_ID")["VALUENUM"].mean().rename(feat)
        vital_agg[feat] = agg

    vitals_df = pd.DataFrame(vital_agg)
    vitals_df.index.name = "ICUSTAY_ID"
    vitals_df = vitals_df.reset_index()

    # ── Aggregate labs per admission ───────────────────────────────
    lab_item_ids = {i for ids in LAB_ITEMS.values() for i in ids}
    lab_filt = labevents[labevents["ITEMID"].isin(lab_item_ids)].copy()
    lab_filt = lab_filt.dropna(subset=["HADM_ID"])
    lab_filt["HADM_ID"] = lab_filt["HADM_ID"].astype(int)

    lab_agg = {}
    for feat, item_ids in LAB_ITEMS.items():
        sub = lab_filt[lab_filt["ITEMID"].isin(item_ids)]
        agg = sub.groupby("HADM_ID")["VALUENUM"].mean().rename(feat)
        lab_agg[feat] = agg

    # Fix glucose: prefer lab glucose over vitals glucose
    if "glucose_lab" in lab_agg:
        lab_agg["glucose"] = lab_agg.pop("glucose_lab")
    else:
        lab_agg.pop("glucose_lab", None)
    labs_df = pd.DataFrame(lab_agg)
    labs_df.index.name = "HADM_ID"
    labs_df = labs_df.reset_index()

    # ── Merge everything per ICU stay ──────────────────────────────
    # icustays already has SUBJECT_ID, so drop it from admissions to avoid
    # SUBJECT_ID_x / SUBJECT_ID_y collision
    adm_cols = ["HADM_ID", "ADMITTIME", "HOSPITAL_EXPIRE_FLAG"]
    if "DISCHTIME" in admissions.columns:
        adm_cols.append("DISCHTIME")
    icu = icu_cardiac.merge(admissions[adm_cols], on="HADM_ID", how="left")
    icu = icu.merge(patients[["SUBJECT_ID", "DOB", "GENDER"]],
                    on="SUBJECT_ID", how="left", suffixes=("", "_pat"))
    icu = icu.merge(vitals_df, on="ICUSTAY_ID", how="left")
    icu = icu.merge(labs_df,   on="HADM_ID",    how="left")
    icu = icu.merge(diag_count, on="HADM_ID",   how="left")

    # ── Derived features ──────────────────────────────────────────
    icu["INTIME"]   = pd.to_datetime(icu["INTIME"],   errors="coerce")
    icu["ADMITTIME"]= pd.to_datetime(icu["ADMITTIME"],errors="coerce")
    icu["DOB"]      = pd.to_datetime(icu["DOB"],      errors="coerce")
    # Use year-only subtraction to avoid int64 overflow
    # (MIMIC sets DOB ~300 years before admission for patients > 89)
    icu["age"] = icu["INTIME"].dt.year - icu["DOB"].dt.year
    icu.loc[icu["age"] > 120, "age"] = 91.4   # MIMIC convention for >89
    icu.loc[icu["age"] < 0, "age"] = np.nan
    icu["los_hours"] = icu["LOS"] * 24   # LOS already in days for MIMIC ICU
    icu["icu_flag"]  = 1
    icu["gender_male"] = (icu["GENDER"] == "M").astype(int)
    icu["days_since_admission"] = (
        (icu["INTIME"] - icu["ADMITTIME"]).dt.total_seconds() / 86400
    ).fillna(0).clip(0)
    icu["prior_mi"]        = icu["SUBJECT_ID"].isin(prior_mi).astype(int)
    icu["prior_hf"]        = icu["SUBJECT_ID"].isin(prior_hf).astype(int)
    icu["prior_arrhythmia"]= icu["SUBJECT_ID"].isin(prior_arr).astype(int)
    icu["apache_score"]    = np.nan   # not available in demo
    icu["hospital_expire_flag"] = icu["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)

    # ── Sort and add visit index ───────────────────────────────────
    icu = icu.sort_values(["SUBJECT_ID", "INTIME"]).reset_index(drop=True)
    icu["visit_index"] = icu.groupby("SUBJECT_ID").cumcount()

    # ── Label: in-hospital death OR deterioration across visits ───
    icu["label"] = icu["hospital_expire_flag"]

    icu["source"]     = "mimic_demo"
    icu["patient_id"] = "mimic_" + icu["SUBJECT_ID"].astype(str)

    # ── Ensure all feature columns exist ──────────────────────────
    for col in FEATURE_COLS:
        if col not in icu.columns:
            icu[col] = np.nan

    result = icu[["patient_id", "visit_index", "label", "source"] + FEATURE_COLS].copy()
    print(f"   ✅ MIMIC arm: {result['patient_id'].nunique()} patients, "
          f"{len(result)} ICU-stay rows, {NUM_FEATURES} features")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# eICU DEMO ARM
# ═══════════════════════════════════════════════════════════════════════════════

def build_eicu_arm() -> pd.DataFrame:
    """Extract temporal cardiac features from eICU Demo."""
    print("\n── eICU Demo Arm ───────────────────────────────────────────")

    patient_f = EICU_DIR / "patient.csv"
    if not patient_f.exists():
        print(f"   ⚠️  Missing eICU files — skipping eICU arm.")
        print(f"       Run: python src/data/download_real_data.py")
        return pd.DataFrame()

    patient = pd.read_csv(patient_f, low_memory=False)
    print(f"   Loaded eICU: {len(patient)} unit stays")

    # Rename for clarity
    patient.columns = [c.lower() for c in patient.columns]

    # ── Filter cardiac diagnoses ────────────────────────────────────
    diag_f = EICU_DIR / "diagnosis.csv"
    if diag_f.exists():
        diag = pd.read_csv(diag_f, low_memory=False)
        diag.columns = [c.lower() for c in diag.columns]
        # Find relevant column
        diag_col = next((c for c in diag.columns if "diagnosisstring" in c or "icd" in c), None)
        if diag_col:
            mask = diag[diag_col].str.lower().str.contains(
                "|".join(CARDIAC_KEYWORDS), na=False
            )
            cardiac_stayids = set(diag[mask]["patientunitstayid"])
        else:
            cardiac_stayids = set(patient["patientunitstayid"])
    else:
        cardiac_stayids = set(patient["patientunitstayid"])

    pat_cardiac = patient[patient["patientunitstayid"].isin(cardiac_stayids)].copy()
    if pat_cardiac.empty:
        print("   ⚠️  No cardiac stays found — using all.")
        pat_cardiac = patient.copy()

    print(f"   Cardiac unit stays: {len(pat_cardiac)}")

    # ── Aggregate periodic vitals per unit stay ────────────────────
    vp_f = EICU_DIR / "vitalPeriodic.csv"
    vital_cols_map = {
        "heartrate":        "heart_rate",
        "systemicsystolic": "sbp",
        "systemicdiastolic":"dbp",
        "respiration":      "resp_rate",
        "temperature":      "temperature",
        "saO2":             "spo2",
    }
    if vp_f.exists():
        vp = pd.read_csv(vp_f, low_memory=False)
        vp.columns = [c.lower() for c in vp.columns]
        stay_ids = set(pat_cardiac["patientunitstayid"])
        vp = vp[vp["patientunitstayid"].isin(stay_ids)]

        col_map_lower = {k.lower(): v for k, v in vital_cols_map.items()}
        rename_dict = {c: col_map_lower[c] for c in vp.columns if c in col_map_lower}
        vp = vp.rename(columns=rename_dict)

        available_vitals = [v for v in vital_cols_map.values() if v in vp.columns]
        vitals_agg = vp.groupby("patientunitstayid")[available_vitals].mean()
    else:
        vitals_agg = pd.DataFrame(index=pat_cardiac["patientunitstayid"])

    # ── Aperiodic vitals (NIBP, glucose) ─────────────────────────
    va_f = EICU_DIR / "vitalAperiodic.csv"
    if va_f.exists():
        va = pd.read_csv(va_f, low_memory=False)
        va.columns = [c.lower() for c in va.columns]
        va = va[va["patientunitstayid"].isin(set(pat_cardiac["patientunitstayid"]))]
        # e.g. noninvasivesystolic, glucosevalue
        aperiodic_cols = {}
        for col in va.columns:
            if "glucose" in col:
                aperiodic_cols[col] = "glucose"
        if aperiodic_cols:
            va_agg = va.groupby("patientunitstayid")[[c for c in aperiodic_cols]].mean()
            va_agg = va_agg.rename(columns=aperiodic_cols)
            vitals_agg = vitals_agg.join(va_agg, how="left")

    # ── Lab values per unit stay ──────────────────────────────────
    lab_col_map = {
        "creatinine": "creatinine", "bun": "bun",
        "potassium":  "potassium",  "sodium": "sodium",
        "hematocrit": "hematocrit", "hemoglobin": "hemoglobin",
        "wbc": "wbc", "platelets": "platelets", "bicarbonate": "bicarbonate",
        "troponin - i": "troponin", "troponin - t": "troponin",
        "bnp": "bnp",
    }
    lab_f = EICU_DIR / "lab.csv"
    if lab_f.exists():
        lab = pd.read_csv(lab_f, low_memory=False)
        lab.columns = [c.lower() for c in lab.columns]
        lab = lab[lab["patientunitstayid"].isin(set(pat_cardiac["patientunitstayid"]))]

        lab_name_col = next((c for c in lab.columns if "labname" in c), None)
        lab_val_col  = next((c for c in lab.columns if "labresult" in c or "value" in c
                             and "labresulttext" not in c), None)

        if lab_name_col and lab_val_col:
            lab["feat"] = lab[lab_name_col].str.lower().str.strip().map(
                {k.lower(): v for k, v in lab_col_map.items()}
            )
            lab_clean = lab.dropna(subset=["feat"])
            lab_clean[lab_val_col] = pd.to_numeric(lab_clean[lab_val_col], errors="coerce")
            labs_agg = (lab_clean.groupby(["patientunitstayid", "feat"])[lab_val_col]
                        .mean().unstack("feat"))
        else:
            labs_agg = pd.DataFrame(index=list(set(pat_cardiac["patientunitstayid"])))
    else:
        labs_agg = pd.DataFrame(index=list(set(pat_cardiac["patientunitstayid"])))

    # ── Apache score ──────────────────────────────────────────────
    apache_f = EICU_DIR / "apachePatientResult.csv"
    if apache_f.exists():
        apache = pd.read_csv(apache_f, low_memory=False)
        apache.columns = [c.lower() for c in apache.columns]
        score_col = next((c for c in apache.columns if "apachescore" in c), None)
        if score_col:
            apache_scores = apache.groupby("patientunitstayid")[score_col].first()
        else:
            apache_scores = pd.Series(dtype=float)
    else:
        apache_scores = pd.Series(dtype=float)

    # ── Assemble per-stay row ─────────────────────────────────────
    df = pat_cardiac.copy()
    df = df.set_index("patientunitstayid")

    # Age
    df["age"] = pd.to_numeric(df.get("age", pd.Series(dtype=str)), errors="coerce")

    # Gender
    gender_col = next((c for c in df.columns if "gender" in c), None)
    df["gender_male"] = (df[gender_col].str.lower() == "male").astype(int) \
        if gender_col else 0

    # LOS
    los_col = next((c for c in df.columns if "unitdischargeoffset" in c or
                    "hospitallosdays" in c or "los" in c), None)
    df["los_hours"] = pd.to_numeric(df[los_col], errors="coerce") if los_col else np.nan
    # eICU los is often in minutes for unit offset
    if los_col and "offset" in los_col:
        df["los_hours"] = df["los_hours"] / 60

    df["icu_flag"]  = 1

    # Join vitals, labs, apache
    if not vitals_agg.empty:
        df = df.join(vitals_agg, how="left")
    if not labs_agg.empty:
        df = df.join(labs_agg, how="left")
    if not apache_scores.empty:
        df = df.join(apache_scores.rename("apache_score"), how="left")
    else:
        df["apache_score"] = np.nan

    # Prior conditions (not available in eICU demo — fill 0)
    df["prior_mi"]         = 0
    df["prior_hf"]         = 0
    df["prior_arrhythmia"] = 0
    df["num_diagnoses"]    = 1  # at least the cardiac diagnosis

    # label: hospital expire flag or similar
    expire_col = next((c for c in df.columns if "expire" in c or "death" in c
                       or "mortality" in c), None)
    df["label"] = pd.to_numeric(df[expire_col], errors="coerce").fillna(0).astype(int) \
        if expire_col else 0
    df["hospital_expire_flag"] = df["label"]

    # Unique patient id (use uniquepid if available)
    uid_col = next((c for c in df.columns if "uniquepid" in c), None)
    if uid_col:
        df["patient_id"] = "eicu_" + df[uid_col].astype(str)
    else:
        df["patient_id"] = "eicu_" + df.index.astype(str)

    # Sort by patient + unit stay offset / index for temporal order
    df = df.reset_index(drop=False)
    df = df.sort_values(["patient_id", "patientunitstayid"]).reset_index(drop=True)
    df["visit_index"] = df.groupby("patient_id").cumcount()

    # Temporal feature: days_since_admission (use hospitaladmitoffset in minutes)
    offset_col = next((c for c in df.columns if "hospitaladmitoffset" in c), None)
    df["days_since_admission"] = (
        pd.to_numeric(df[offset_col], errors="coerce") / 1440 if offset_col  # min → days
        else 0
    )
    df["days_since_admission"] = df["days_since_admission"].fillna(0).clip(0)

    df["source"] = "eicu_demo"

    # ── Select only shared feature cols ──────────────────────────
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    result = df[["patient_id", "visit_index", "label", "source"] + FEATURE_COLS].copy()
    print(f"   ✅ eICU arm: {result['patient_id'].nunique()} patients, "
          f"{len(result)} unit-stay rows, {NUM_FEATURES} features")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FUSION + PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_and_preprocess(mimic_df: pd.DataFrame, eicu_df: pd.DataFrame):
    """Combine both arms, preprocess, save arrays."""

    parts = [df for df in [mimic_df, eicu_df] if not df.empty]
    if not parts:
        print("❌ Both arms empty — check downloads.")
        sys.exit(1)

    combined = pd.concat(parts, ignore_index=True)
    print(f"\n── Fusion ──────────────────────────────────────────────────")
    print(f"   Total rows  : {len(combined):,}")
    print(f"   Total patients: {combined['patient_id'].nunique():,}")
    print(f"   Sources     : {combined['source'].value_counts().to_dict()}")
    print(f"   Labels      : pos={combined['label'].sum()} "
          f"neg={(combined['label']==0).sum()} "
          f"rate={combined['label'].mean():.1%}")

    # Ensure numeric — handle duplicate columns from merges
    # First deduplicate columns (keep first occurrence)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    for col in FEATURE_COLS:
        if col in combined.columns:
            s = combined[col]
            # If column is a DataFrame (duplicated), take first
            if hasattr(s, 'columns'):
                s = s.iloc[:, 0]
            combined[col] = pd.to_numeric(s, errors="coerce")
        else:
            combined[col] = np.nan

    # Forward-fill within patient, then global median
    print("\n   Imputing missing values...")
    combined = combined.sort_values(["patient_id", "visit_index"])
    for col in FEATURE_COLS:
        combined[col] = (combined.groupby("patient_id")[col]
                         .transform(lambda x: x.ffill().bfill()))
    med = combined[FEATURE_COLS].median()
    combined[FEATURE_COLS] = combined[FEATURE_COLS].fillna(med)

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(REAL_DIR / "real_longitudinal_features.csv", index=False)
    print(f"   💾 Saved fused CSV → {REAL_DIR / 'real_longitudinal_features.csv'}")

    # ── Build LSTM sequences (sliding window) ──────────────────────
    SEQ_LEN = 3   # Use seq_len=3 since demo data has few visits per patient
    PRED_HORIZON = 1

    X_list, y_list = [], []
    for pid, grp in combined.groupby("patient_id"):
        grp = grp.sort_values("visit_index").reset_index(drop=True)
        vals = grp[FEATURE_COLS].values.astype(np.float32)
        labels = grp["label"].values

        if len(grp) < SEQ_LEN + PRED_HORIZON:
            # Pad with first row if patient has fewer visits than SEQ_LEN
            pad_needed = SEQ_LEN + PRED_HORIZON - len(grp)
            vals   = np.vstack([np.tile(vals[[0]], (pad_needed, 1)), vals])
            labels = np.concatenate([[labels[0]] * pad_needed, labels])

        for i in range(len(vals) - SEQ_LEN - PRED_HORIZON + 1):
            X_list.append(vals[i: i + SEQ_LEN])
            y_list.append(int(labels[i + SEQ_LEN + PRED_HORIZON - 1]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"\n   Sequences created: {len(y):,}  shape: {X.shape}")

    if len(y) < 10:
        print("❌  Too few sequences. Check data download.")
        sys.exit(1)

    # ── Split ──────────────────────────────────────────────────────
    try:
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.15 / 0.85, random_state=42, stratify=y_tv)
    except ValueError:
        # Fall back without stratify if class too small
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv,
                                                           test_size=0.15/0.85,
                                                           random_state=42)

    # ── Scale ──────────────────────────────────────────────────────
    n_tr, T, F = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(n_tr, T, F)
    X_val   = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)

    # ── Save ───────────────────────────────────────────────────────
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREP_DIR / "X_train.npy", X_train)
    np.save(PREP_DIR / "X_val.npy",   X_val)
    np.save(PREP_DIR / "X_test.npy",  X_test)
    np.save(PREP_DIR / "y_train.npy", y_train)
    np.save(PREP_DIR / "y_val.npy",   y_val)
    np.save(PREP_DIR / "y_test.npy",  y_test)
    joblib.dump(scaler,       PREP_DIR / "preprocessor.pkl")
    joblib.dump(FEATURE_COLS, PREP_DIR / "feature_names.pkl")

    print(f"\n📊 Final Preprocessed Arrays:")
    print(f"   X_train : {X_train.shape}  y_train : {y_train.shape}  "
          f"pos={y_train.sum()}")
    print(f"   X_val   : {X_val.shape}    y_val   : {y_val.shape}    "
          f"pos={y_val.sum()}")
    print(f"   X_test  : {X_test.shape}   y_test  : {y_test.shape}   "
          f"pos={y_test.sum()}")
    print(f"\n💾 Saved to: {PREP_DIR}")


def main():
    print("\n" + "=" * 60)
    print("  Building Real Clinical Dataset")
    print("  MIMIC-III Demo + eICU Demo")
    print("=" * 60)

    mimic_df = build_mimic_arm()
    eicu_df  = build_eicu_arm()
    fuse_and_preprocess(mimic_df, eicu_df)

    print("\n✅ Real dataset build complete!")
    print("   Next step:")
    print("     python src/models/train_lstm.py")
    print("     python src/models/train_gru.py")
    print("     python src/models/train_transformer.py")


if __name__ == "__main__":
    main()
