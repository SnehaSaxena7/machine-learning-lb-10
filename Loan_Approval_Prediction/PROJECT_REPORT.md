# Project Report: Loan Approval Prediction System

## 1. Abstract
This project demonstrates an end-to-end supervised classification workflow using Python and Scikit-learn. It covers data loading, preprocessing, model comparison, evaluation, persistence and a simple Streamlit interface.

## 2. Problem Statement
Build a reproducible machine-learning prototype that accepts structured input and returns a model-generated class prediction.

## 3. Objectives
- Perform data preprocessing.
- Compare multiple classification algorithms.
- Evaluate models using suitable metrics.
- Save the trained model.
- Develop a simple prediction interface.

## 4. Tools and Technologies
Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib and Streamlit.

## 5. Methodology
CSV ingestion → exploratory inspection → train/test split → imputation/encoding/scaling → model training → metric comparison → best pipeline persistence → UI prediction.

## 6. Algorithms
Logistic Regression, Random Forest Classifier and Gradient Boosting Classifier.

## 7. Evaluation
Accuracy, weighted precision, weighted recall, weighted F1-score, classification report and confusion matrix. Actual scores are generated locally by `python train_model.py`; see `model_comparison.csv`.

## 8. Results
Record the measured scores from `model_comparison.csv` after running training. Do not report assumed accuracy values.

## 9. Limitations and Ethics
The supplied data are synthetic and the model is an educational prototype. It must not be used for real-world eligibility or health decisions.

## 10. Conclusion
The project demonstrates the main stages of an end-to-end ML application and produces a saved model pipeline and an interactive prediction interface.
