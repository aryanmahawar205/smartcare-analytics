"""Machine learning pipeline creation, training, evaluation, threshold tuning, and feature importance."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

logger = logging.getLogger(__name__)

# Try importing XGBoost safely
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost is not installed or available.")


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer for numerical and categorical features."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    return preprocessor


def get_model_pipeline(model_type: str, numeric_cols: list, categorical_cols: list, random_state: int = 42) -> Pipeline:
    """Construct a full Pipeline with preprocessor and classifier."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    if model_type == "logistic_regression":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is requested but not available in this environment.")
        # Note: XGBoost scale_pos_weight for imbalance ~ (neg / pos) = 0.886 / 0.114 ~ 7.8
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=7.8,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate pipeline on test data and compute performance metrics."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    prec_array, rec_array, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_array, prec_array)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "fpr": fpr,
        "tpr": tpr,
        "prec_array": prec_array,
        "rec_array": rec_array,
        "threshold": threshold,
    }


def find_optimal_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    metric: str = "f1"
) -> float:
    """Find threshold that optimizes F1 or PR-AUC on cross-validation predictions on training set."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    cv_probs = cross_val_predict(
        pipeline, X_train, y_train, groups=groups, cv=sgkf, method="predict_proba"
    )[:, 1]

    best_threshold = 0.5
    best_score = -1.0

    thresholds = np.linspace(0.1, 0.9, 81)
    for th in thresholds:
        preds = (cv_probs >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_train, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_train, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_train, preds, zero_division=0)
        else:
            score = f1_score(y_train, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = th

    return float(best_threshold)


def get_feature_importances(pipeline: Pipeline, numeric_cols: list, categorical_cols: list, top_n: int = 20) -> pd.DataFrame:
    """Extract feature importances or coefficients from trained pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame()

    feat_imp = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    return feat_imp
