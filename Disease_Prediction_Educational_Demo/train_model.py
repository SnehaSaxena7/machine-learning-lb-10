"""Train and compare classification models; run from this project directory."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "synthetic_symptom_dataset.csv"
TARGET = "label"
NUMERIC = ['age', 'symptom_duration_days', 'fever', 'cough', 'sore_throat', 'runny_nose', 'sneezing', 'itchy_eyes', 'headache', 'nausea', 'vomiting', 'diarrhea', 'body_ache', 'fatigue', 'light_sensitivity']
CATEGORICAL = ['sex']

def main():
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    results, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)
        }
        results.append(row)
        fitted[name] = pipe
        print(f"{name}: accuracy={row['accuracy']:.3f}, F1={row['f1_weighted']:.3f}")
    result_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    result_df.to_csv(ROOT / "model_comparison.csv", index=False)
    best_name = result_df.iloc[0]["model"]
    best = fitted[best_name]
    preds = best.predict(X_test)
    report = classification_report(y_test, preds, zero_division=0)
    (ROOT / "classification_report.txt").write_text(
        f"Best model: {best_name}\\n\\n{report}", encoding="utf-8"
    )
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix - " + best_name)
    plt.tight_layout(); plt.savefig(ROOT / "confusion_matrix.png", dpi=160); plt.close()
    joblib.dump(best, ROOT / "best_model.joblib")
    (ROOT / "model_metadata.json").write_text(json.dumps({
        "best_model": best_name,
        "target": TARGET,
        "features": list(X.columns),
        "test_size": 0.20,
        "random_state": 42
    }, indent=2), encoding="utf-8")
    print("Saved best_model.joblib, metrics, report and confusion_matrix.png")

if __name__ == "__main__":
    main()
