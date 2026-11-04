# System Flowchart

```text
Start
  |
  v
Load CSV Dataset
  |
  v
Inspect / Clean Data
  |
  v
Train-Test Split (80:20)
  |
  v
Preprocessing Pipeline
(Imputation, Encoding, Scaling)
  |
  v
Train 3 Classification Models
  |
  v
Compare Evaluation Metrics
  |
  v
Select Best F1 Model
  |
  v
Save Pipeline with Joblib
  |
  v
Streamlit User Interface
  |
  v
Display Demo Prediction
  |
  v
End
```

## System Architecture
User → Streamlit UI → Saved Scikit-learn Pipeline (preprocessing + classifier) → Predicted class → UI result.
Training pipeline: CSV → Pandas → preprocessing → candidate estimators → evaluation → Joblib artifact.
