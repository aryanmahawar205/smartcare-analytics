"""Plotly visualization helpers for healthcare analytics and model evaluation."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg

# Standard healthcare analytics color palette
PRIMARY_COLOR = "#005a9c"
SECONDARY_COLOR = "#00a896"
ACCENT_COLOR = "#e63946"
NEUTRAL_COLOR = "#f4f1de"


def plot_readmission_distribution(df: pd.DataFrame):
    """Plot readmission target distribution."""
    counts = df["readmitted_30d"].value_counts().reset_index()
    counts.columns = ["Readmitted within 30 days", "Count"]
    counts["Label"] = counts["Readmitted within 30 days"].map({1: "Yes (<30d)", 0: "No (>=30d / None)"})

    fig = px.bar(
        counts,
        x="Label",
        y="Count",
        color="Label",
        color_discrete_map={"Yes (<30d)": ACCENT_COLOR, "No (>=30d / None)": PRIMARY_COLOR},
        title="30-Day Hospital Readmission Distribution",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of Encounters")
    return fig


def plot_demographic_readmission(df: pd.DataFrame, feature: str = "age"):
    """Plot readmission rates across demographic categories (age, race, gender)."""
    if feature not in df.columns:
        return None

    grouped = df.groupby(feature, observed=False)["readmitted_30d"].agg(["count", "mean"]).reset_index()
    grouped.columns = [feature, "Encounter Count", "Readmission Rate"]
    grouped["Readmission Rate (%)"] = grouped["Readmission Rate"] * 100

    fig = px.bar(
        grouped,
        x=feature,
        y="Readmission Rate (%)",
        text="Readmission Rate (%)",
        title=f"30-Day Readmission Rate by {feature.capitalize()}",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_title="30-Day Readmission Rate (%)", xaxis_title=feature.capitalize())
    return fig


def plot_utilization_boxplots(df: pd.DataFrame, metric: str = "number_inpatient"):
    """Plot distribution of prior healthcare utilization by readmission status."""
    df_plot = df.copy()
    df_plot["Readmission Status"] = df_plot["readmitted_30d"].map({1: "Readmitted (<30d)", 0: "Not Readmitted"})

    fig = px.box(
        df_plot,
        x="Readmission Status",
        y=metric,
        color="Readmission Status",
        color_discrete_map={"Readmitted (<30d)": ACCENT_COLOR, "Not Readmitted": PRIMARY_COLOR},
        title=f"{metric.replace('_', ' ').capitalize()} Distribution by Readmission Status",
    )
    fig.update_layout(xaxis_title="", yaxis_title=metric.replace("_", " ").capitalize())
    return fig


def plot_roc_curves(eval_results: dict):
    """Plot ROC curves comparing multiple models."""
    fig = gg.Figure()

    for model_name, eval_dict in eval_results.items():
        fig.add_trace(gg.Scatter(
            x=eval_dict["fpr"],
            y=eval_dict["tpr"],
            mode="lines",
            name=f"{model_name} (AUC = {eval_dict['roc_auc']:.3f})"
        ))

    # Add diagonal chance line
    fig.add_trace(gg.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random Chance"
    ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(x=0.6, y=0.1)
    )
    return fig


def plot_pr_curves(eval_results: dict):
    """Plot Precision-Recall curves comparing multiple models."""
    fig = gg.Figure()

    for model_name, eval_dict in eval_results.items():
        fig.add_trace(gg.Scatter(
            x=eval_dict["rec_array"],
            y=eval_dict["prec_array"],
            mode="lines",
            name=f"{model_name} (PR-AUC = {eval_dict['pr_auc']:.3f})"
        ))

    fig.update_layout(
        title="Precision-Recall Curve Comparison",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        legend=dict(x=0.6, y=0.8)
    )
    return fig


def plot_confusion_matrix_heatmap(cm: np.ndarray, model_name: str = "Model"):
    """Plot confusion matrix heatmap."""
    labels = ["No (<30d)", "Yes (<30d)"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {model_name}"
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
    )
    return fig


def plot_top_feature_importances(feat_imp_df: pd.DataFrame, model_name: str = "Model"):
    """Plot bar chart of top feature importances."""
    if feat_imp_df.empty:
        return None

    feat_sorted = feat_imp_df.sort_values(by="importance", ascending=True)

    fig = px.bar(
        feat_sorted,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top Feature Importances / Predictors — {model_name}",
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    fig.update_layout(xaxis_title="Importance Score", yaxis_title="Feature")
    return fig
