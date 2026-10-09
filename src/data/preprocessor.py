"""Data cleaning, filtering, target creation, ICD-9 mapping, and leakage-free splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Expired / Hospice discharge disposition IDs where readmission is impossible
EXPIRED_HOSPICE_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]


def map_icd9_category(code: str) -> str:
    """Map ICD-9 diagnosis code to high-level clinical category."""
    if pd.isna(code) or code == "?" or str(code).strip() == "":
        return "Missing"

    code_str = str(code).strip()

    # Handle codes starting with 'V' or 'E'
    if code_str.startswith("V"):
        return "Supplementary_V"
    if code_str.startswith("E"):
        return "External_E"

    try:
        val = float(code_str)
    except ValueError:
        return "Other"

    # Clinical ICD-9 ranges
    # Check 250.xx diabetes first or integer floor 250
    if int(val) == 250:
        return "Diabetes"
    elif (390 <= val <= 459) or val == 785:
        return "Circulatory"
    elif (460 <= val <= 519) or val == 786:
        return "Respiratory"
    elif (520 <= val <= 579) or val == 787:
        return "Digestive"
    elif (800 <= val <= 999):
        return "Injury"
    elif (710 <= val <= 739):
        return "Musculoskeletal"
    elif (580 <= val <= 629) or val == 788:
        return "Genitourinary"
    elif (140 <= val <= 239):
        return "Neoplasms"
    elif (780 <= val <= 799):
        return "Symptoms"
    elif (240 <= val <= 279):
        return "Endocrine/Metabolic"
    else:
        return "Other"


def clean_and_preprocess_raw_data(df: pd.DataFrame, filter_expired: bool = True) -> pd.DataFrame:
    """Clean raw dataset: replace '?' with NaN, remove expired/hospice cases, map ICD-9 codes, create target.

    Args:
        df: Raw DataFrame.
        filter_expired: Whether to filter out expired / hospice discharge encounters.

    Returns:
        Cleaned DataFrame with 'readmitted_30d' binary target column.
    """
    cleaned = df.copy()

    # Replace '?' with NaN across all columns
    cleaned = cleaned.replace("?", np.nan)

    # Filter out expired or hospice encounters if specified
    if filter_expired and "discharge_disposition_id" in cleaned.columns:
        cleaned = cleaned[~cleaned["discharge_disposition_id"].isin(EXPIRED_HOSPICE_DISPOSITION_IDS)].copy()

    # Create binary target: 1 if readmitted <30 days, else 0
    if "readmitted" in cleaned.columns:
        cleaned["readmitted_30d"] = (cleaned["readmitted"].astype(str).str.strip() == "<30").astype(int)

    # Map diagnosis ICD-9 codes to categories
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in cleaned.columns:
            cleaned[f"{diag_col}_cat"] = cleaned[diag_col].apply(map_icd9_category)

    return cleaned


def get_feature_and_target_cols():
    """Return categorical, numerical, and excluded column names for modelling."""
    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    categorical_cols = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "payer_code",
        "diag_1_cat",
        "diag_2_cat",
        "diag_3_cat",
        "max_glu_serum",
        "A1Cresult",
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "change",
        "diabetesMed",
    ]

    # Excluded from predictors: identifiers, raw readmitted label, raw ICD-9 codes
    excluded_cols = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "diag_1",
        "diag_2",
        "diag_3",
        "weight",  # >96% missing in dataset
    ]

    return numeric_cols, categorical_cols, excluded_cols


def split_data_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Perform a patient-level train/test split ensuring no patient_nbr overlap.

    Args:
        df: Cleaned DataFrame with 'patient_nbr' and 'readmitted_30d'.
        test_size: Fraction of data for test set.
        random_state: Fixed seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    num_cols, cat_cols, _ = get_feature_and_target_cols()
    feature_cols = [c for c in num_cols + cat_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["readmitted_30d"].copy()
    groups = df["patient_nbr"].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    # Verification checks
    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])
    overlap = train_patients.intersection(test_patients)
    assert len(overlap) == 0, f"Patient leakage detected! Overlapping patients: {len(overlap)}"

    return X_train, X_test, y_train, y_test
