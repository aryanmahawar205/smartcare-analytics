# SmartCare Analytics

### AI-Powered Hospital Readmission Risk Prediction & Healthcare Analytics Platform

SmartCare Analytics is a machine-learning-based healthcare analytics platform designed to explore patient hospitalization data, understand readmission patterns, and estimate the risk of early hospital readmission (within 30 days of discharge).

The project combines rigorous healthcare data preprocessing, exploratory data analysis, predictive modelling, model evaluation, and interactive Plotly visualizations in a polished multi-page Streamlit web application.

---

## Executive Summary & Healthcare Context

Unplanned hospital readmissions within 30 days of discharge represent a major quality metric and financial driver in modern healthcare delivery. High 30-day readmission rates often signify gaps in inpatient care transitions, post-discharge planning, or outpatient follow-up.

### Target Variable Definition
- **Positive Class (`1`)**: Readmitted within 30 days (`readmitted == '<30'`).
- **Negative Class (`0`)**: Not readmitted within 30 days (`readmitted == '>30'` or `readmitted == 'NO'`).

### Clinical Data Cleaning & Inclusion Criteria
Encounters involving patient death or transfer to hospice care (`discharge_disposition_id` in `11, 13, 14, 19, 20, 21`) are explicitly excluded from readmission risk analysis, as readmission is clinically impossible for these outcomes.

### Data Leakage Prevention
Patient identifiers (`patient_nbr`) and encounter identifiers (`encounter_id`) are strictly excluded from predictive features. All train/test dataset splitting is performed at the **patient level** using `GroupShuffleSplit` on `patient_nbr` to ensure that encounters from the same patient never appear in both training and evaluation sets. Cross-validation and decision threshold tuning are performed exclusively on the training folds using `StratifiedGroupKFold`.

---

## Application Architecture & Repository Structure

```text
smartcare-analytics/
├── app.py                      # Main Streamlit application entry point & page router
├── requirements.txt            # Python dependencies (Streamlit, Pandas, Scikit-learn, Plotly, Pytest, XGBoost)
├── README.md                   # Complete project documentation & execution instructions
├── AGENTS.md                   # Agent guidelines & project requirements
├── src/                        # Core modular Python packages
│   ├── data/
│   │   ├── loader.py           # Automated UCI dataset retrieval & extraction
│   │   └── preprocessor.py     # ICD-9 mapping, target creation, leakage-free group split
│   ├── models/
│   │   └── pipeline.py         # Scikit-learn pipelines, cross-validation threshold tuning, evaluation metrics
│   └── visualization/
│       └── plots.py            # Plotly analytics & diagnostic evaluation charts
├── views/                      # Streamlit navigation page modules
│   ├── overview.py             # Executive summary & key statistics
│   ├── explorer.py             # Interactive healthcare data exploration & filters
│   ├── models.py               # Model evaluation comparison, ROC/PR curves & feature importances
│   └── risk_explorer.py        # Sample risk scoring & batch CSV prediction tool
└── tests/                      # Automated Pytest suite
    ├── test_loader.py          # Dataset download unit tests
    ├── test_preprocessor.py    # Data cleaning, target, and group split leakage checks
    └── test_pipeline.py        # ML pipeline execution & evaluation metric tests
```

---

## Machine Learning Pipeline & Performance Summary

The project evaluates three classification models on a patient-separated test set (~19,800 encounters):

1. **Logistic Regression (Baseline)**: Balanced class weights, maximum 1000 lbfgs iterations.
2. **Random Forest Classifier**: Ensemble classifier with 150 trees, max depth 12, balanced class weights.
3. **XGBoost Classifier (Optional)**: Gradient boosting with `scale_pos_weight` class balancing.

### Key Evaluation Metrics (Patient-Separated Test Set)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6673 | 0.2203 | 0.2077 | 0.4328 | 0.2807 | 0.55 |
| **Random Forest** | 0.6677 | 0.2162 | 0.1877 | 0.5262 | 0.2767 | 0.51 |
| **XGBoost** | 0.6716 | 0.2327 | 0.2047 | 0.4368 | 0.2787 | 0.55 |

### Primary Predictive Factors
Across all models, the top features associated with 30-day readmission risk include:
1. **Prior Healthcare Utilization**: Number of prior inpatient admissions (`number_inpatient`) and emergency visits (`number_emergency`).
2. **Discharge Disposition**: Transfers to Skilled Nursing Facilities (SNF) or home health services versus direct routine home discharge.
3. **Inpatient Complexity**: Number of medications administered (`num_medications`) and hospital length of stay (`time_in_hospital`).

---

## Dataset Attribution & Source

The primary dataset used in this project is the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository:

- **Source URL**: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Citation**: Strack, B., DeShazo, J. P., Ojeda, C., Higginson, C. A., Konduri, S. R., Xie, A., & Clore, J. N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records. *BioMed Research International*, 2014.

---

## Local Installation & Execution Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
```bash
PYTHONPATH=. pytest
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will automatically download and extract the dataset to `data/raw/` on first launch and cache model training results using Streamlit decorators (`@st.cache_data` and `@st.cache_resource`).

---

## Educational Disclaimer

**SmartCare Analytics is an educational healthcare analytics prototype.**

All predictions, probability estimates, and risk categories are statistical outputs generated from historical hospital encounter data for academic exploration. They do **not** constitute clinical advice, medical diagnoses, treatment recommendations, or validated patient risk scores for real-world medical practice.
