# src/predict.py
from pathlib import Path

import joblib
import pandas as pd

ART_DIR = Path("artifacts") / "model_data"
MODEL_PATH = ART_DIR / "xgb_pipeline_raw.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_survival_proba(raw_input: dict) -> float:
    """
    raw_input должен содержать колонки:
    Sex, Age, Fare, Pclass, SibSp, Parch, Cabin, Title
    """
    model = load_model()

    X = pd.DataFrame([{
        "Sex": raw_input.get("Sex"),
        "Age": raw_input.get("Age"),
        "Fare": raw_input.get("Fare"),
        "Pclass": raw_input.get("Pclass"),
        "SibSp": raw_input.get("SibSp"),
        "Parch": raw_input.get("Parch"),
        "Cabin": raw_input.get("Cabin"),
        "Title": raw_input.get("Title"),
    }])

    proba = model.predict_proba(X)[0, 1]  # P(Survived=1)
    return float(proba)


if __name__ == "__main__":
    example = {
        "Sex": "female",
        "Age": 28,
        "Fare": 30.0,
        "Pclass": 2,
        "SibSp": 0,
        "Parch": 0,
        "Cabin": "",        
        "Title": "Mrs",     
    }

    p = predict_survival_proba(example)
    print(f"Survival probability: {p:.4f} ({p*100:.1f}%)")
