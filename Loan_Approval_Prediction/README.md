# Loan Approval Prediction System

## Overview
A complete educational machine-learning classification project for MCA Capstone Lab Sheet-10.

## Dataset
- `loan_approval_dataset.csv` — included CSV dataset.
- Target column: `Loan_Status`.
- The dataset is synthetic and included for learning/testing; it is not an official institutional or commercial dataset.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Workflow
1. Load CSV data using Pandas.
2. Impute missing values; scale numeric features; one-hot encode categorical features.
3. Split data into 80% training and 20% test partitions with stratification.
4. Compare Logistic Regression, Random Forest and Gradient Boosting.
5. Evaluate accuracy, weighted precision, weighted recall and weighted F1.
6. Save the best complete preprocessing/model pipeline with Joblib.
7. Use the Streamlit interface for a demonstration prediction.

## Files generated after training
- `best_model.joblib`
- `model_comparison.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `model_metadata.json`

## Important note
This project predicts a synthetic loan-status label for coursework only. It is not a lending tool and must not be used to approve or reject real applicants.

## Limitations
Synthetic data cannot establish real-world model performance. Results are only meaningful for demonstrating the ML workflow. Do not use this model for real decisions.
