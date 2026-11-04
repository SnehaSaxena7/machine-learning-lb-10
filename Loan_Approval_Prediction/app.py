import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.joblib"

st.set_page_config(page_title="Loan Approval Demo", page_icon="📊")
st.title("Loan Approval Prediction System")
st.warning("Educational prototype only. This synthetic-data model must not be used for actual lending or financial decisions.")

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
with st.form("loan_form"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_emp = st.selectbox("Self employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant income", min_value=0, value=5000)
    with c2:
        co_income = st.number_input("Co-applicant income", min_value=0, value=1500)
        loan_amount = st.number_input("Loan amount (dataset units)", min_value=1, value=150)
        loan_term = st.selectbox("Loan term (months)", [120, 180, 240, 300, 360])
        credit = st.selectbox("Credit history", [1, 0])
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("Predict")
if submitted:
    row = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_emp,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit, "Property_Area": property_area
    }])
    pred = int(model.predict(row)[0])
    st.subheader("Demo model output")
    st.success("Likely Approved (class 1)" if pred == 1 else "Likely Not Approved (class 0)")
    st.caption("This is a model classification output, not a credit decision or guarantee.")
