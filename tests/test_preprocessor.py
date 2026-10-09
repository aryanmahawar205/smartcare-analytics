"""Unit tests for preprocessing, target creation, expired/hospice filtering, and leakage-free splitting."""

import pandas as pd
import numpy as np
from src.data.preprocessor import (
    map_icd9_category,
    clean_and_preprocess_raw_data,
    get_feature_and_target_cols,
    split_data_by_patient,
    EXPIRED_HOSPICE_DISPOSITION_IDS
)


def test_map_icd9_category():
    assert map_icd9_category("414.01") == "Circulatory"
    assert map_icd9_category("250.02") == "Diabetes"
    assert map_icd9_category("250") == "Diabetes"
    assert map_icd9_category("486") == "Respiratory"
    assert map_icd9_category("V45") == "Supplementary_V"
    assert map_icd9_category("E812") == "External_E"
    assert map_icd9_category("?") == "Missing"
    assert map_icd9_category(None) == "Missing"


def test_clean_and_preprocess_raw_data():
    sample_df = pd.DataFrame({
        "encounter_id": [1, 2, 3, 4],
        "patient_nbr": [101, 102, 103, 104],
        "race": ["Caucasian", "?", "AfricanAmerican", "Asian"],
        "discharge_disposition_id": [1, 11, 1, 13],  # 11 and 13 are expired/hospice
        "readmitted": ["<30", ">30", "NO", "<30"],
        "diag_1": ["250.01", "410", "?", "V58"],
    })

    cleaned = clean_and_preprocess_raw_data(sample_df, filter_expired=True)

    # Should filter out IDs 11 and 13 (rows with encounter 2 and 4)
    assert len(cleaned) == 2
    assert set(cleaned["encounter_id"]) == {1, 3}

    # Check target creation
    assert "readmitted_30d" in cleaned.columns
    assert cleaned.loc[cleaned["encounter_id"] == 1, "readmitted_30d"].values[0] == 1
    assert cleaned.loc[cleaned["encounter_id"] == 3, "readmitted_30d"].values[0] == 0

    # Check '?' replaced by NaN
    assert not pd.isna(cleaned.loc[cleaned["encounter_id"] == 1, "race"].values[0])
    assert cleaned.loc[cleaned["encounter_id"] == 1, "diag_1_cat"].values[0] == "Diabetes"
    assert cleaned.loc[cleaned["encounter_id"] == 3, "diag_1_cat"].values[0] == "Missing"


def test_patient_level_leakage_free_split():
    df = pd.DataFrame({
        "patient_nbr": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        "encounter_id": range(1, 21),
        "time_in_hospital": [1] * 20,
        "num_lab_procedures": [10] * 20,
        "num_procedures": [1] * 20,
        "num_medications": [5] * 20,
        "number_outpatient": [0] * 20,
        "number_emergency": [0] * 20,
        "number_inpatient": [0] * 20,
        "number_diagnoses": [3] * 20,
        "race": ["Caucasian"] * 20,
        "gender": ["Female"] * 20,
        "age": ["[50-60)"] * 20,
        "admission_type_id": [1] * 20,
        "discharge_disposition_id": [1] * 20,
        "admission_source_id": [7] * 20,
        "medical_specialty": ["Missing"] * 20,
        "payer_code": ["Missing"] * 20,
        "diag_1_cat": ["Diabetes"] * 20,
        "diag_2_cat": ["Circulatory"] * 20,
        "diag_3_cat": ["Respiratory"] * 20,
        "max_glu_serum": ["None"] * 20,
        "A1Cresult": ["None"] * 20,
        "metformin": ["No"] * 20,
        "repaglinide": ["No"] * 20,
        "nateglinide": ["No"] * 20,
        "chlorpropamide": ["No"] * 20,
        "glimepiride": ["No"] * 20,
        "acetohexamide": ["No"] * 20,
        "glipizide": ["No"] * 20,
        "glyburide": ["No"] * 20,
        "tolbutamide": ["No"] * 20,
        "pioglitazone": ["No"] * 20,
        "rosiglitazone": ["No"] * 20,
        "acarbose": ["No"] * 20,
        "miglitol": ["No"] * 20,
        "troglitazone": ["No"] * 20,
        "tolazamide": ["No"] * 20,
        "examide": ["No"] * 20,
        "citoglipton": ["No"] * 20,
        "insulin": ["No"] * 20,
        "glyburide-metformin": ["No"] * 20,
        "glipizide-metformin": ["No"] * 20,
        "glimepiride-pioglitazone": ["No"] * 20,
        "metformin-rosiglitazone": ["No"] * 20,
        "metformin-pioglitazone": ["No"] * 20,
        "change": ["No"] * 20,
        "diabetesMed": ["Yes"] * 20,
        "readmitted_30d": [0, 1] * 10
    })

    X_train, X_test, y_train, y_test = split_data_by_patient(df, test_size=0.2, random_state=42)

    # Identifiers must not be in X columns
    num_cols, cat_cols, excluded_cols = get_feature_and_target_cols()
    for col in excluded_cols:
        assert col not in X_train.columns
        assert col not in X_test.columns

    # Verify NO patient overlap
    train_patients = set(df.loc[X_train.index, "patient_nbr"])
    test_patients = set(df.loc[X_test.index, "patient_nbr"])
    assert len(train_patients.intersection(test_patients)) == 0
