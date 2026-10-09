"""View 3: Model Performance & Comparison."""

import streamlit as st
import pandas as pd

from src.visualization.plots import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix_heatmap,
    plot_top_feature_importances,
)


def render_models_view(eval_results: dict, trained_models: dict, feature_importances: dict):
    st.header("Model Performance & Comparison")
    st.markdown("""
    Compare predictive performance across baseline Logistic Regression, Random Forest, and optional XGBoost models evaluated on the patient-separated test set.
    """)

    # Performance metrics table
    st.subheader("Model Evaluation Summary (Test Set)")
    summary_data = []
    for model_name, eval_dict in eval_results.items():
        summary_data.append({
            "Model": model_name,
            "ROC-AUC": f"{eval_dict['roc_auc']:.4f}",
            "PR-AUC": f"{eval_dict['pr_auc']:.4f}",
            "Precision": f"{eval_dict['precision']:.4f}",
            "Recall": f"{eval_dict['recall']:.4f}",
            "F1-Score": f"{eval_dict['f1_score']:.4f}",
            "Decision Threshold": f"{eval_dict['threshold']:.2f}",
        })
    st.table(pd.DataFrame(summary_data))

    st.divider()

    # ROC and PR Curves
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_roc_curves(eval_results), use_container_width=True)
    with col2:
        st.plotly_chart(plot_pr_curves(eval_results), use_container_width=True)

    st.divider()

    # Model selector for confusion matrix and feature importances
    selected_model = st.selectbox("Select Model for Detailed Diagnostic Inspection", list(eval_results.keys()))

    if selected_model in eval_results:
        eval_dict = eval_results[selected_model]
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                plot_confusion_matrix_heatmap(eval_dict["confusion_matrix"], model_name=selected_model),
                use_container_width=True
            )
        with col4:
            if selected_model in feature_importances:
                st.plotly_chart(
                    plot_top_feature_importances(feature_importances[selected_model], model_name=selected_model),
                    use_container_width=True
                )

    st.markdown("""
    ### Key Clinical Analytics Findings
    1. **Prior Healthcare Utilization**: Prior inpatient admissions (`number_inpatient`) and emergency visits (`number_emergency`) are consistently the top predictors of 30-day readmission.
    2. **Discharge Disposition**: Discharge to SNF (Skilled Nursing Facility) or home health is associated with elevated readmission risk compared to direct home discharge.
    3. **Precision vs. Recall**: In readmission risk management, high recall (sensitivity) ensures high-risk patients are flagged for transition-of-care interventions.
    """)
