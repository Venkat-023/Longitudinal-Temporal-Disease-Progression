# Dataset Documentation

## Dataset Name

This project uses **MIMIC-IV 2.1**, a large real-world critical care and hospital
electronic health record dataset. The local raw file used by the pipeline is:

```text
data/mimic_iv_raw/mimic-iv-2-1.zip
```

The dataset is stored as a ZIP file and the pipeline reads the CSV files inside
the ZIP directly. It does not extract the full uncompressed dataset to disk.

## Prediction Problem

The task is **cardiac progression prediction** for ICU patients.

For each patient sequence, the model predicts whether the patient's cardiac
condition is likely to worsen.

The target is binary classification:

```text
label = 1: worsening cardiac progression
label = 0: stable or improving cardiac progression
```

This is a progression classification task, not a regression task. The neural
networks output one logit per sequence. After sigmoid activation, that logit
becomes a probability of worsening.

## Raw Tables Used

The preprocessing script reads the following MIMIC-IV tables:

```text
hosp/patients.csv
hosp/admissions.csv
hosp/diagnoses_icd.csv
icu/icustays.csv
icu/chartevents.csv
hosp/labevents.csv
```

### patients.csv

Used for patient-level demographic context:

- `subject_id`
- `gender`
- `anchor_age`
- `anchor_year`
- `dod`

### admissions.csv

Used for hospital admission context and mortality fallback labels:

- `subject_id`
- `hadm_id`
- `admittime`
- `dischtime`
- `admission_type`
- `hospital_expire_flag`

### diagnoses_icd.csv

Used to identify the cardiac cohort and prior cardiac history:

- `subject_id`
- `hadm_id`
- `icd_code`
- `icd_version`

The cardiac cohort is built from ICD-9 and ICD-10 prefixes for ischemic heart
disease, myocardial infarction, heart failure, arrhythmia, cardiomyopathy,
valvular disease, hypertensive heart disease, pulmonary heart disease, and other
heart disease groups.

### icustays.csv

Used to build the longitudinal ICU visit timeline:

- `subject_id`
- `hadm_id`
- `stay_id`
- `intime`
- `outtime`
- `los`

### chartevents.csv

This is the large ICU bedside charting table. It is streamed in chunks because
it is too large to load fully into memory. The pipeline keeps only cardiac ICU
stays and selected vital-sign item IDs.

Extracted vital groups:

- Heart rate
- Systolic blood pressure
- Diastolic blood pressure
- Mean blood pressure
- Respiratory rate
- Temperature
- SpO2
- Fingerstick glucose

### labevents.csv

This is the hospital laboratory events table. It is also streamed in chunks.
The pipeline keeps only cardiac admissions and selected lab item IDs.

Extracted lab groups:

- Troponin T
- Troponin I
- BNP
- NT-proBNP
- Creatinine
- BUN
- Lab glucose
- Potassium
- Sodium
- Chloride
- Bicarbonate
- Hematocrit
- Hemoglobin
- WBC
- Platelets

## Cohort Construction

The pipeline first builds a cardiac cohort:

1. Load patients, admissions, diagnoses, and ICU stays.
2. Mark diagnoses as cardiac using ICD-9 and ICD-10 prefixes.
3. Select ICU stays where the subject or admission belongs to the cardiac cohort.
4. Merge patient, admission, ICU, and diagnosis-count metadata.
5. Add prior-condition flags:
   - `prior_mi`
   - `prior_hf`
   - `prior_arrhythmia`
6. Add temporal context:
   - `days_since_last_admission`
   - `time_delta_days`
   - `visit_index`

In the current full MIMIC-IV run, the cohort stage detected:

```text
Cardiac subjects   : 66,841
Cardiac admissions : 157,019
Cardiac ICU stays  : 52,130
```

## Feature Engineering

The final feature vector has **59 features per ICU stay**.

For vital signs, the pipeline aggregates repeated measurements per ICU stay.
For labs, the pipeline aggregates repeated measurements per hospital admission.

Common aggregation patterns:

- `mean`: average measured value
- `std`: within-stay or within-admission variability
- `max`: maximum measured value for clinically meaningful labs

Temperature values below 50 are treated as Celsius and converted to Fahrenheit.
Physiologically impossible lab values are clipped to reasonable ranges before
aggregation.

## Final Feature List

### Vital Signs

```text
heart_rate_mean
heart_rate_std
sbp_mean
sbp_std
dbp_mean
dbp_std
mbp_mean
mbp_std
resp_rate_mean
resp_rate_std
temperature_mean
temperature_std
spo2_mean
spo2_std
glucose_fingerstick_mean
glucose_fingerstick_std
```

### Cardiac Markers

```text
troponin_t_mean
troponin_t_max
troponin_i_mean
troponin_i_max
bnp_mean
bnp_max
nt_probnp_mean
nt_probnp_max
```

### Metabolic and Renal Features

```text
creatinine_mean
creatinine_max
bun_mean
bun_max
glucose_lab_mean
glucose_lab_std
```

### Electrolytes

```text
potassium_mean
potassium_std
sodium_mean
sodium_std
chloride_mean
chloride_std
bicarbonate_mean
bicarbonate_std
```

### Complete Blood Count

```text
hematocrit_mean
hematocrit_std
hemoglobin_mean
hemoglobin_std
wbc_mean
wbc_max
platelets_mean
platelets_std
```

### Clinical Context

```text
age
los_days
num_procedures
prior_mi
prior_hf
prior_arrhythmia
num_medications
icu_flag
```

### Admission-Level Context

```text
admission_type_emergency
admission_type_elective
days_since_last_admission
```

### Temporal Context

```text
time_delta_days
visit_index
```

## Cardiac Severity Index

The pipeline computes a **Cardiac Severity Index (CSI)** per ICU stay.

CSI is a hand-engineered clinical severity score. Higher CSI means more severe
cardiac condition. It is computed from:

- Heart rate deviation from 60-100 bpm
- Systolic blood pressure outside expected range
- Troponin T elevation
- BNP or NT-proBNP elevation
- SpO2 reduction
- Creatinine elevation
- Respiratory rate elevation
- Hemoglobin reduction

CSI is used to build the progression label.

## Label Construction

Patients are sorted chronologically by ICU admission time.

For a patient with a next ICU stay:

```text
label = 1 if next_csi > current_csi * 1.10
label = 0 otherwise
```

That means the patient is labeled as worsening if the next ICU stay has more
than a 10 percent increase in CSI.

For a patient's last ICU stay, or for patients with only one stay, the pipeline
uses hospital mortality:

```text
label = hospital_expire_flag
```

## Missing Values

The pipeline handles missing values in two stages:

1. Forward-fill and backward-fill within each patient timeline.
2. Fill any remaining missing values using the global median of each feature.

This keeps the temporal structure while still ensuring every model receives a
complete numeric tensor.

## Sequence Construction

The models do not train on a single ICU stay. They train on fixed-length
temporal sequences.

Default sequence length:

```text
seq_len = 6
```

Each model input sample has shape:

```text
(6, 59)
```

Meaning:

- 6 ICU visits or padded visit positions
- 59 engineered features per timestep

If a patient has fewer than 6 ICU stays, the earliest available row is repeated
as left-padding. This lets every patient contribute a fixed-shape sequence.

For patients with at least 6 stays, the pipeline creates sliding windows.

## Train, Validation, and Test Split

After sequences are built, the data is split into:

- 70 percent training
- 15 percent validation
- 15 percent testing

The split uses `train_test_split` with `random_state=42`. Stratified splitting
is attempted first so the worsening/stable label ratio is preserved. If
stratification is impossible because of class counts, it falls back to ordinary
random splitting.

## Scaling

The pipeline uses `StandardScaler`.

Important detail:

- The scaler is fit only on the training set.
- Validation and test sets are transformed using the training scaler.

The 3D tensor is temporarily reshaped from:

```text
(samples, timesteps, features)
```

to:

```text
(samples * timesteps, features)
```

for scaling, then reshaped back.

## Saved Outputs

The preprocessing script saves:

```text
data/real/mimiciv_longitudinal_features.csv
data/preprocessed/X_train.npy
data/preprocessed/X_val.npy
data/preprocessed/X_test.npy
data/preprocessed/y_train.npy
data/preprocessed/y_val.npy
data/preprocessed/y_test.npy
data/preprocessed/preprocessor.pkl
data/preprocessed/feature_names.pkl
```

## Model-Ready Tensor Format

Each model receives:

```text
X: float32 tensor of shape (batch_size, 6, 59)
y: float32 or int label of shape (batch_size,)
```

The model output is:

```text
logit: one scalar per sequence
probability: sigmoid(logit)
class: probability >= 0.5 means worsening
```
