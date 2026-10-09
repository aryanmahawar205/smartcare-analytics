"""View 2: Healthcare Data Explorer."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.visualization.plots import (
    plot_readmission_distribution,
    plot_demographic_readmission,
    plot_utilization_boxplots,
)


def render_explorer_view(df: pd.DataFrame):
    st.header("Healthcare Data Explorer")
    st.markdown("Interactively explore demographics, admission characteristics, diagnoses, and healthcare utilization patterns.")

    st.sidebar.subheader("Data Explorer Filters")
    age_filter = st.sidebar.multiselect(
        "Select Age Groups",
        options=sorted(df["age"].dropna().unique().tolist()),
        default=sorted(df["age"].dropna().unique().tolist())
    )

    race_filter = st.sidebar.multiselect(
        "Select Race",
        options=sorted(df["race"].dropna().unique().tolist()),
        default=sorted(df["race"].dropna().unique().tolist())
    )

    # Filter dataframe
    filtered_df = df[df["age"].isin(age_filter) & df["race"].isin(race_filter)].copy()

    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** encounters based on sidebar filters.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Readmission & Demographics",
        "Clinical & Utilization",
        "Diagnoses & Medications",
        "Data Quality & Missing Values"
    ])

    with tab1:
        st.subheader("30-Day Readmission Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_readmission_distribution(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_demographic_readmission(filtered_df, feature="age"), use_container_width=True)

        st.subheader("Readmission Rate by Race and Gender")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_demographic_readmission(filtered_df, feature="race"), use_container_width=True)
        with col4:
            st.plotly_chart(plot_demographic_readmission(filtered_df, feature="gender"), use_container_width=True)

    with tab2:
        st.subheader("Healthcare Utilization Patterns")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_utilization_boxplots(filtered_df, metric="number_inpatient"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_utilization_boxplots(filtered_df, metric="number_emergency"), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_utilization_boxplots(filtered_df, metric="num_medications"), use_container_width=True)
        with col4:
            st.plotly_chart(plot_utilization_boxplots(filtered_df, metric="time_in_hospital"), use_container_width=True)

    with tab3:
        st.subheader("Primary Diagnosis (ICD-9 Category) vs Readmission")
        if "diag_1_cat" in filtered_df.columns:
            diag_grouped = filtered_df.groupby("diag_1_cat")["readmitted_30d"].agg(["count", "mean"]).reset_index()
            diag_grouped.columns = ["Primary Diagnosis Category", "Encounters", "Readmission Rate"]
            diag_grouped["Readmission Rate (%)"] = diag_grouped["Readmission Rate"] * 100

            fig_diag = px.bar(
                diag_grouped.sort_values(by="Readmission Rate (%)", ascending=False),
                x="Primary Diagnosis Category",
                y="Readmission Rate (%)",
                text="Readmission Rate (%)",
                color="Encounters",
                title="30-Day Readmission Rate by Primary Diagnosis Category",
            )
            fig_diag.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_diag, use_container_width=True)

        st.subheader("Diabetes Medication Changes & Readmission")
        if "change" in filtered_df.columns:
            change_grouped = filtered_df.groupby("change")["readmitted_30d"].agg(["count", "mean"]).reset_index()
            change_grouped["Readmission Rate (%)"] = change_grouped["mean"] * 100
            fig_change = px.bar(
                change_grouped,
                x="change",
                y="Readmission Rate (%)",
                text="Readmission Rate (%)",
                title="Readmission Rate by Diabetes Medication Change Status (Ch vs No)",
                color_discrete_sequence=["#00a896"]
            )
            fig_change.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_change, use_container_width=True)

    with tab4:
        st.subheader("Data Quality & Missing Values Summary")
        missing_df = df.isna().sum().reset_index()
        missing_df.columns = ["Feature", "Missing Count"]
        missing_df["Missing Percentage (%)"] = (missing_df["Missing Count"] / len(df)) * 100
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)

        if not missing_df.empty:
            st.dataframe(missing_df, hide_index=True, use_container_width=True)
            fig_missing = px.bar(
                missing_df,
                x="Feature",
                y="Missing Percentage (%)",
                text="Missing Percentage (%)",
                title="Missing Value Percentage by Feature",
                color_discrete_sequence=["#e63946"]
            )
            fig_missing.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values found after initial cleaning!")
