import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.joblib"
SYMPTOMS = ["fever","cough","sore_throat","runny_nose","sneezing","itchy_eyes",
            "headache","nausea","vomiting","diarrhea","body_ache","fatigue","light_sensitivity"]

st.set_page_config(page_title="Symptom Classifier Demo", page_icon="🧪")
st.title("Disease Label Classification — Educational Demo")
st.error("Not a medical device. It uses synthetic teaching data and cannot diagnose, rule out, or recommend treatment for any condition. For health concerns, consult a qualified healthcare professional.")
if not MODEL_PATH.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
with st.form("symptom_form"):
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    sex = st.selectbox("Sex", ["Female","Male","Other"])
    duration = st.number_input("Symptom duration (days)", min_value=1, max_value=60, value=2)
    st.write("Select symptoms reported (demo input):")
    selected = {}
    cols = st.columns(3)
    for i, symptom in enumerate(SYMPTOMS):
        with cols[i % 3]:
            selected[symptom] = int(st.checkbox(symptom.replace("_"," ").title()))
    submitted = st.form_submit_button("Classify demo input")
if submitted:
    row = pd.DataFrame([{"age":age, "sex":sex, "symptom_duration_days":duration, **selected}])
    pred = model.predict(row)[0]
    st.subheader("Synthetic model label")
    st.info(f"Model class: {pred}")
    st.caption("This output is not a diagnosis. Do not make medical decisions based on it.")
