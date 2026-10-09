# Agent Instructions — SmartCare Analytics

## 1. Project Goal

Build SmartCare Analytics from scratch as an educational machine-learning and healthcare analytics project.

The application must focus on meaningful healthcare data analysis and predictive modelling, with a polished and easy-to-use interface.

Prioritize correctness, explainability, usability, reproducibility, and clear documentation over unnecessary complexity.

## 2. Scope and Constraints

- Use Python as the primary language.
- Use Streamlit for the user interface.
- Use Pandas and NumPy for data preparation.
- Use Scikit-learn for the machine-learning pipeline.
- Use Plotly for interactive visualizations.
- XGBoost may be included if it can be integrated reliably.
- Keep the application runnable locally on a typical student laptop.
- Do not add cloud services, cloud storage, authentication, payment systems, microservices, or external AI APIs.
- Do not introduce a database unless there is a clear and necessary reason.
- Avoid unnecessary dependencies and excessive abstraction.
- Keep the code modular, readable, and straightforward to maintain.

## 3. Dataset

Use the official UCI Diabetes 130-US Hospitals dataset:

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The data represents hospital encounters for patients diagnosed with diabetes.

Implement a reproducible way to download and load the raw dataset into a locally ignored data directory. If automatic retrieval fails, provide clear manual-download instructions.

Do not commit the full dataset or generated model artifacts.

Handle missing values and invalid or unknown categorical values explicitly. Do not silently replace missing clinical values with misleading values.

Credit the UCI Machine Learning Repository and cite the dataset in the README.

## 4. Prediction Objective

The primary task is binary classification of early hospital readmission.

Create the target as follows:

- `readmitted == "<30"` → positive class (1).
- `readmitted == ">30"` or `readmitted == "NO"` → negative class (0).

Normalize categorical labels safely before constructing the target.

Clearly explain that the negative class means no readmission within 30 days, not necessarily no readmission ever.

Do not use the original `readmitted` column as a predictor.

Never use `encounter_id` or `patient_nbr` as predictive features. If the raw patient identifier is available, use it only to prevent the same patient's encounters from appearing in both training and test sets.

Use a reproducible, patient-group-aware train/test split where possible. If this is not feasible with the available data interface, investigate a raw-data loading solution rather than silently introducing patient-level leakage.

Fit preprocessing transformations only on training data. Keep preprocessing and model estimation together in Scikit-learn pipelines wherever appropriate.

## 5. Machine Learning

Build a manageable, reliable classification pipeline.

At minimum:

1. Establish a Logistic Regression baseline.
2. Train and evaluate a Random Forest classifier.
3. Include XGBoost only if it is practical and does not make the project fragile.
4. Account for class imbalance where appropriate.
5. Set fixed random seeds for reproducibility.
6. Compare the models on the same evaluation split.

Evaluate using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-recall AUC (PR-AUC)
- Confusion matrix
- Classification report

Do not use accuracy as the only performance measure.

Display the evaluation results from actual model runs. Never fabricate model scores, improvement percentages, training results, or research findings.

Document the selected model and explain the reasons for selecting it.

## 6. Explainability and Responsible Analytics

Provide interpretable insights through feature importance or permutation importance.

SHAP may be included if it integrates cleanly and can be explained clearly. It is not mandatory if it introduces unnecessary complexity.

Explain that feature importance and associations do not establish causation.

Where practical, inspect model performance across relevant demographic subgroups and note limitations, small sample sizes, and potential bias.

Do not describe predictions as diagnoses, medical advice, or definitive clinical risk assessments.

## 7. User Interface

Create a polished, responsive Streamlit application with a consistent visual style.

Include the following logical sections, using pages or navigation as appropriate:

### Overview
- Project introduction and healthcare use case.
- Key dataset and readmission statistics.
- Clear explanation of the prediction target.

### Healthcare Data Explorer
- Dataset size and feature summaries.
- Missing-value analysis.
- Readmission distribution.
- Useful demographic, diagnosis, admission, and healthcare-utilization visualizations.
- Filters where helpful.

### Model Performance
- Comparison of model metrics.
- Confusion matrix.
- ROC and precision-recall curves.
- Feature importance and interpretation.

### Readmission Risk Explorer
- Display predicted probabilities and risk groupings for eligible encounter records.
- Support CSV-based batch scoring if practical.
- Clearly validate expected input columns.
- Make it possible to inspect and download prediction results.

Do not claim that a patient's score is clinically validated. Clearly label all risk categories as illustrative and educational.

The interface should have helpful empty states, progress messages, and readable error messages. Avoid placeholder charts or fabricated statistics.

## 8. Engineering Quality

Use a simple, understandable structure that separates data loading, preprocessing, modelling, evaluation, and UI logic.

- Provide a `requirements.txt` with only necessary dependencies.
- Add automated tests for target creation, preprocessing, leakage prevention, and core model-pipeline behavior.
- Use reusable functions instead of duplicated logic.
- Handle missing files and download failures gracefully.
- Avoid retraining expensive models unnecessarily on every Streamlit rerun.
- Keep raw data, processed data, cached files, and trained artifacts out of Git.
- Ensure that the application does not require API keys or secrets.

## 9. Documentation and Verification

Keep the README accurate and complete.

Document:

- The healthcare problem and dataset source.
- The meaning of the target variable.
- Setup and execution instructions.
- The modelling workflow and evaluation metrics.
- Major design decisions and limitations.
- The project's educational, non-clinical nature.

Run the available tests and perform practical verification of the application.

Fix errors where possible, and report any verification that could not be completed. Do not claim that an untested feature works.

## 10. Completion Criteria

The project is complete when:

- The dataset can be loaded through documented steps.
- The preprocessing and training pipeline runs successfully.
- The classifiers can be evaluated reproducibly.
- The dashboard shows real dataset statistics and model results.
- Predictions can be explored and downloaded where supported.
- The core automated tests pass.
- The README accurately explains how to run the project.
- No cloud infrastructure or credentials are required.

Keep the implementation focused. Deliver a solid healthcare analytics project rather than a general-purpose hospital management system.
