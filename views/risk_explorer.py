"""View 4: Readmission Risk Explorer & Batch Scoring."""

import streamlit as st
import pandas as pd
import numpy as np

from src.data.preprocessor import clean_and_preprocess_raw_data, get_feature_and_target_cols


def render_risk_explorer_view(df: pd.DataFrame, trained_models: dict, eval_results: dict):
    st.header("Readmission Risk Explorer & Batch Scoring")
    st.markdown("""
    Scoring tool to calculate predicted 30-day readmission probabilities for individual encounter samples or CSV uploads using trained models.
    """)

    available_models = list(trained_models.keys())
    if not available_models:
        st.error("No trained models available.")
        return

    selected_model_name = st.selectbox("Select Model for Risk Scoring", available_models)
    model = trained_models[selected_model_name]
    threshold = eval_results[selected_model_name]["threshold"] if selected_model_name in eval_results else 0.5

    st.caption(f"Using **{selected_model_name}** with classification decision threshold = **{threshold:.2f}**")

    tab1, tab2 = st.tabs(["Interactive Sample Explorer", "CSV Batch Scoring"])

    with tab1:
        st.subheader("Sample Encounter Risk Prediction")
        st.markdown("Select a sample encounter record from the cleaned dataset to estimate readmission risk:")

        sample_idx = st.slider("Sample Encounter Index", min_value=0, max_value=min(len(df) - 1, 5000), value=10)
        sample_row = df.iloc[[sample_idx]].copy()

        num_cols, cat_cols, _ = get_feature_and_target_cols()
        feature_cols = [c for c in num_cols + cat_cols if c in sample_row.columns]

        X_sample = sample_row[feature_cols]
        prob = model.predict_proba(X_sample)[0, 1]
        is_high_risk = prob >= threshold

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Readmission Risk", f"{prob * 100:.1f}%")
        col2.metric("Risk Level Category", "HIGH RISK (<30d)" if is_high_risk else "LOW/MODERATE RISK")

        actual_label = sample_row["readmitted_30d"].values[0] if "readmitted_30d" in sample_row.columns else None
        col3.metric("Actual Ground Truth Label", "Readmitted (<30d)" if actual_label == 1 else "Not Readmitted (<30d)")

        st.subheader("Encounter Key Characteristics")
        st.dataframe(
            sample_row[["age", "race", "gender", "time_in_hospital", "num_medications", "number_inpatient", "number_emergency", "number_diagnoses", "diag_1_cat"]],
            use_container_width=True
        )

    with tab2:
        st.subheader("CSV Batch Prediction & Download")
        st.markdown("""
        Upload a CSV file containing patient encounter records to score readmission risk in batch.
        The uploaded CSV should contain the standard UCI Diabetes dataset features.
        """)

        uploaded_file = st.file_uploader("Upload Encounter CSV File", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_raw = pd.read_csv(uploaded_file)
                st.write(f"Uploaded **{len(batch_raw):,}** rows.")

                batch_clean = clean_and_preprocess_raw_data(batch_raw, filter_expired=False)
                X_batch = batch_clean[feature_cols]

                batch_probs = model.predict_proba(X_batch)[:, 1]
                batch_preds = (batch_probs >= threshold).astype(int)

                results_df = batch_raw.copy()
                results_df["predicted_readmission_probability"] = np.round(batch_probs, 4)
                results_df["predicted_risk_flag"] = np.where(batch_preds == 1, "High Risk (<30d)", "Low/Moderate Risk")

                st.subheader("Batch Prediction Results Sample")
                st.dataframe(results_df[["encounter_id", "patient_nbr", "predicted_readmission_probability", "predicted_risk_flag"]].head(20), use_container_width=True)

                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Prediction Results CSV",
                    data=csv_data,
                    file_name="readmission_risk_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
        else:
            st.info("Upload a dataset CSV file above to run batch predictions.")
