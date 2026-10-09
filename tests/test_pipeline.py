"""Unit tests for ML pipeline building, evaluation, threshold finding, and feature importances."""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import get_feature_and_target_cols
from src.models.pipeline import (
    get_model_pipeline,
    evaluate_model,
    find_optimal_threshold,
    get_feature_importances,
    XGBOOST_AVAILABLE
)


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "patient_nbr": np.random.choice(range(1, 50), n),
        "time_in_hospital": np.random.randint(1, 14, n),
        "num_lab_procedures": np.random.randint(10, 80, n),
        "num_procedures": np.random.randint(0, 6, n),
        "num_medications": np.random.randint(1, 30, n),
        "number_outpatient": np.random.randint(0, 5, n),
        "number_emergency": np.random.randint(0, 5, n),
        "number_inpatient": np.random.randint(0, 5, n),
        "number_diagnoses": np.random.randint(1, 9, n),
        "race": np.random.choice(["Caucasian", "AfricanAmerican", "Asian", None], n),
        "gender": np.random.choice(["Female", "Male"], n),
        "age": np.random.choice(["[50-60)", "[60-70)", "[70-80)"], n),
        "admission_type_id": np.random.choice([1, 2, 3], n),
        "discharge_disposition_id": np.random.choice([1, 3, 6], n),
        "admission_source_id": np.random.choice([1, 7], n),
        "medical_specialty": np.random.choice(["Cardiology", "InternalMedicine", "Missing"], n),
        "payer_code": np.random.choice(["MC", "HM", "Missing"], n),
        "diag_1_cat": np.random.choice(["Circulatory", "Diabetes", "Respiratory"], n),
        "diag_2_cat": np.random.choice(["Circulatory", "Diabetes", "Other"], n),
        "diag_3_cat": np.random.choice(["Circulatory", "Other"], n),
        "max_glu_serum": np.random.choice(["None", "Norm", ">200"], n),
        "A1Cresult": np.random.choice(["None", "Norm", ">8"], n),
        "metformin": np.random.choice(["No", "Steady"], n),
        "repaglinide": ["No"] * n,
        "nateglinide": ["No"] * n,
        "chlorpropamide": ["No"] * n,
        "glimepiride": ["No"] * n,
        "acetohexamide": ["No"] * n,
        "glipizide": ["No"] * n,
        "glyburide": ["No"] * n,
        "tolbutamide": ["No"] * n,
        "pioglitazone": ["No"] * n,
        "rosiglitazone": ["No"] * n,
        "acarbose": ["No"] * n,
        "miglitol": ["No"] * n,
        "troglitazone": ["No"] * n,
        "tolazamide": ["No"] * n,
        "examide": ["No"] * n,
        "citoglipton": ["No"] * n,
        "insulin": np.random.choice(["No", "Steady", "Up", "Down"], n),
        "glyburide-metformin": ["No"] * n,
        "glipizide-metformin": ["No"] * n,
        "glimepiride-pioglitazone": ["No"] * n,
        "metformin-rosiglitazone": ["No"] * n,
        "metformin-pioglitazone": ["No"] * n,
        "change": np.random.choice(["No", "Ch"], n),
        "diabetesMed": np.random.choice(["Yes", "No"], n),
        "readmitted_30d": np.random.choice([0, 1], n, p=[0.85, 0.15])
    })
    return df


def test_logistic_regression_pipeline(dummy_data):
    num_cols, cat_cols, _ = get_feature_and_target_cols()
    X = dummy_data[num_cols + cat_cols]
    y = dummy_data["readmitted_30d"]

    pipe = get_model_pipeline("logistic_regression", num_cols, cat_cols)
    pipe.fit(X, y)

    eval_results = evaluate_model(pipe, X, y, threshold=0.5)
    assert 0.0 <= eval_results["roc_auc"] <= 1.0
    assert 0.0 <= eval_results["pr_auc"] <= 1.0
    assert eval_results["confusion_matrix"].shape == (2, 2)


def test_random_forest_pipeline(dummy_data):
    num_cols, cat_cols, _ = get_feature_and_target_cols()
    X = dummy_data[num_cols + cat_cols]
    y = dummy_data["readmitted_30d"]

    pipe = get_model_pipeline("random_forest", num_cols, cat_cols)
    pipe.fit(X, y)

    feat_imp = get_feature_importances(pipe, num_cols, cat_cols, top_n=10)
    assert not feat_imp.empty
    assert "feature" in feat_imp.columns
    assert "importance" in feat_imp.columns
    assert len(feat_imp) <= 10


def test_xgboost_pipeline(dummy_data):
    if not XGBOOST_AVAILABLE:
        pytest.skip("XGBoost not available")

    num_cols, cat_cols, _ = get_feature_and_target_cols()
    X = dummy_data[num_cols + cat_cols]
    y = dummy_data["readmitted_30d"]

    pipe = get_model_pipeline("xgboost", num_cols, cat_cols)
    pipe.fit(X, y)

    eval_results = evaluate_model(pipe, X, y, threshold=0.5)
    assert "roc_auc" in eval_results
