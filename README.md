# SmartCare Analytics

### AI-Powered Hospital Readmission Risk Prediction & Healthcare Analytics Platform

SmartCare Analytics is a machine-learning-based healthcare analytics project designed to explore patient hospitalization data, understand readmission patterns, and estimate the risk of early hospital readmission.

The project combines exploratory data analysis, predictive modelling, model evaluation, and interactive visualizations in a simple web application.

## Objectives

- Analyse healthcare data to identify readmission patterns and trends.
- Develop machine learning models to predict hospital readmission within 30 days.
- Compare model performance using appropriate classification metrics.
- Identify important factors associated with readmission predictions.
- Present healthcare insights through an interactive analytics dashboard.
- Discuss model limitations, bias, and implications for healthcare analytics.

## Dataset

The primary dataset will be the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

Official source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

It contains hospital encounter records for patients diagnosed with diabetes, including demographic information, diagnoses, laboratory procedures, medications, and previous healthcare utilization.

The original dataset contains three readmission labels:

- `<30`: Readmitted within 30 days.
- `>30`: Readmitted after 30 days.
- `No`: No recorded readmission.

The primary prediction task is binary classification: identify encounters associated with readmission within 30 days. The other two labels will be treated as the negative class, meaning **not readmitted within 30 days**, rather than implying that the patient was never readmitted.

The original dataset must be appropriately attributed in the project documentation.

## Planned Features

- Healthcare data loading, cleaning, and preprocessing.
- Exploratory data analysis and descriptive statistics.
- Readmission distribution and patient-demographic visualizations.
- Machine learning model training and comparison.
- Evaluation using precision, recall, F1-score, ROC-AUC, PR-AUC, and a confusion matrix.
- Feature importance and interpretable model insights.
- Interactive healthcare analytics dashboard.
- Risk scoring for eligible patient encounter records.
- Downloadable prediction results and analytics reports, where practical.

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn and, where practical, XGBoost
- **Visualization:** Plotly
- **User interface:** Streamlit
- **Testing:** Pytest

## Local Execution

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The implementation should document any required dataset preparation or download steps and provide clear instructions for running the application locally.

## Important Disclaimer

SmartCare Analytics is an educational healthcare analytics prototype. Predictions are estimates generated from historical data and must not be treated as clinical advice, diagnoses, or recommendations for real patient care.

The dataset reflects historical healthcare practices at selected US hospitals and may not generalize to other populations, hospitals, or current clinical practice.
