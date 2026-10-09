"""SmartCare Analytics — Streamlit Main Application Entry Point."""

import streamlit as st
import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessor import clean_and_preprocess_raw_data, split_data_by_patient, get_feature_and_target_cols
from src.models.pipeline import get_model_pipeline, evaluate_model, find_optimal_threshold, get_feature_importances, XGBOOST_AVAILABLE

from views.overview import render_overview_view
from views.explorer import render_explorer_view
from views.models import render_models_view
from views.risk_explorer import render_risk_explorer_view

# Streamlit Page Config
st.set_page_config(
    page_title="SmartCare Analytics — Hospital Readmission AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading and cleaning UCI Diabetes dataset...")
def load_and_preprocess_dataset():
    df_raw = load_raw_data()
    df_clean = clean_and_preprocess_raw_data(df_raw, filter_expired=True)
    return df_clean


@st.cache_resource(show_spinner="Training machine learning models and preparing evaluations...")
def train_and_evaluate_all_models(df_clean: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data_by_patient(df_clean)
    num_cols, cat_cols, _ = get_feature_and_target_cols()

    groups_train = df_clean.loc[X_train.index, "patient_nbr"]

    trained_models = {}
    eval_results = {}
    feature_importances = {}

    model_types = ["logistic_regression", "random_forest"]
    if XGBOOST_AVAILABLE:
        model_types.append("xgboost")

    for m_type in model_types:
        name_map = {
            "logistic_regression": "Logistic Regression Baseline",
            "random_forest": "Random Forest Classifier",
            "xgboost": "XGBoost Classifier",
        }
        display_name = name_map[m_type]

        pipe = get_model_pipeline(m_type, num_cols, cat_cols)
        pipe.fit(X_train, y_train)

        best_threshold = find_optimal_threshold(pipe, X_train, y_train, groups=groups_train, n_splits=3, metric="f1")
        eval_dict = evaluate_model(pipe, X_test, y_test, threshold=best_threshold)

        feat_imp = get_feature_importances(pipe, num_cols, cat_cols, top_n=20)

        trained_models[display_name] = pipe
        eval_results[display_name] = eval_dict
        feature_importances[display_name] = feat_imp

    return trained_models, eval_results, feature_importances


def main():
    st.sidebar.title("🏥 SmartCare Analytics")
    st.sidebar.caption("AI-Powered Hospital Readmission Risk Platform")

    page = st.sidebar.radio(
        "Navigation Pages",
        [
            "Overview & Context",
            "Healthcare Data Explorer",
            "Model Performance & Metrics",
            "Readmission Risk Explorer",
        ]
    )

    # Load cached dataset
    try:
        df_clean = load_and_preprocess_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Load cached trained models
    try:
        trained_models, eval_results, feature_importances = train_and_evaluate_all_models(df_clean)
    except Exception as e:
        st.error(f"Error training machine learning models: {e}")
        st.stop()

    # Route views
    if page == "Overview & Context":
        render_overview_view(df_clean)
    elif page == "Healthcare Data Explorer":
        render_explorer_view(df_clean)
    elif page == "Model Performance & Metrics":
        render_models_view(eval_results, trained_models, feature_importances)
    elif page == "Readmission Risk Explorer":
        render_risk_explorer_view(df_clean, trained_models, eval_results)


if __name__ == "__main__":
    main()
