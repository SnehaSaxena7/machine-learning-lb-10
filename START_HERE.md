# MCA Capstone Projects — Start Here

This ZIP contains both Lab Sheet-10 project options:
1. Loan Approval Prediction System
2. Disease Prediction System (educational symptom-label demo)

Each project folder contains source code, a synthetic CSV dataset, Streamlit app, report, flowchart, viva questions, GitHub checklist, and a 13-slide PowerPoint.

## Run either project
1. Open the relevant project folder in VS Code or a terminal.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train_model.py`
4. Start the UI: `streamlit run app.py`

Training creates `best_model.joblib`, `model_comparison.csv`, `classification_report.txt`, `confusion_matrix.png`, and `model_metadata.json`.

**Important:** Both included datasets are synthetic and for academic demonstration only. The loan project is not for real credit decisions. The disease project is not a diagnostic tool or a substitute for professional healthcare.
