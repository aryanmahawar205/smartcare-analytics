"""View 1: Overview & Healthcare Context."""

import streamlit as st
import pandas as pd


def render_overview_view(df: pd.DataFrame):
    st.header("Hospital Readmission Analytics Overview")

    st.markdown("""
    ### Project Background & Healthcare Use Case
    Unplanned hospital readmissions within 30 days of discharge are a critical quality metric and major cost driver in healthcare.
    High readmission rates often indicate gaps in inpatient care transition, discharge planning, or outpatient follow-up.

    **SmartCare Analytics** is an educational healthcare machine-learning platform designed to explore patient hospitalization records,
    identify key risk factors associated with early hospital readmission, and deliver actionable analytics for care teams.
    """)

    st.divider()

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    total_encounters = len(df)
    unique_patients = df["patient_nbr"].nunique() if "patient_nbr" in df.columns else total_encounters
    readmitted_30d = df["readmitted_30d"].sum() if "readmitted_30d" in df.columns else 0
    readmit_rate = (readmitted_30d / total_encounters) * 100 if total_encounters > 0 else 0

    col1.metric("Total Encounters", f"{total_encounters:,}")
    col2.metric("Unique Patients", f"{unique_patients:,}")
    col3.metric("30-Day Readmissions", f"{readmitted_30d:,}")
    col4.metric("30-Day Readmission Rate", f"{readmit_rate:.2f}%")

    st.divider()

    st.subheader("Dataset & Clinical Context")
    st.markdown("""
    * **Data Source**: UCI Machine Learning Repository — *Diabetes 130-US Hospitals for Years 1999–2008* dataset.
    * **Target Definition**: Binary classification where `readmitted == '<30'` is the **positive class (1)**, representing early readmission within 30 days. Encounters with `>30` or `NO` readmission form the **negative class (0)**.
    * **Inclusion/Exclusion Criteria**: Encounters involving patient death or transfer to hospice care (`discharge_disposition_id` 11, 13, 14, 19, 20, 21) are excluded from readmission analysis as readmission is not possible.
    * **Data Integrity**: Identifiers (`encounter_id`, `patient_nbr`) are strictly excluded from predictive features, and patient-level splitting (`GroupShuffleSplit`) ensures no patient data leakage between train and evaluation sets.
    """)

    st.info("⚠️ **Educational Disclaimer**: SmartCare Analytics is a non-clinical academic prototype. Risk scores and metrics are illustrative statistical estimates based on historical data and do not constitute clinical recommendations or diagnoses.")
