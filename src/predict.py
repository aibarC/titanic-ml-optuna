# src/predict.py
from pathlib import Path
import json

import joblib
import pandas as pd

ART_DIR = Path("artifacts") / "model_data"
MODEL_PATH = ART_DIR / "xgb_pipeline_raw.joblib"
THRESH_PATH = ART_DIR / "threshold.json"

_model = None
_threshold = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_threshold() -> float:
    global _threshold
    if _threshold is None:
        with open(THRESH_PATH, "r", encoding="utf-8") as f:
            _threshold = float(json.load(f)["threshold_f1"])
    return _threshold


def predict(raw_input: dict) -> dict:
    """
    raw_input keys expected:
    Sex, Age, Fare, Pclass, SibSp, Parch, Cabin, Title

    returns:
      {
        "proba": float,
        "percent": "69.0%",
        "threshold": float,
        "pred": 0/1
      }
    """
    model = get_model()
    thr = get_threshold()

    X = pd.DataFrame([{
        "Sex": raw_input.get("Sex"),
        "Age": raw_input.get("Age"),
        "Fare": raw_input.get("Fare"),
        "Pclass": raw_input.get("Pclass"),
        "SibSp": raw_input.get("SibSp"),
        "Parch": raw_input.get("Parch"),
        "Cabin": raw_input.get("Cabin", ""),
        "Title": raw_input.get("Title"),
    }])

    proba = float(model.predict_proba(X)[0, 1])
    return {
        "proba": proba,
        "percent": f"{proba * 100:.1f}%",
        "threshold_f1": thr,
        "pred": int(proba >= thr),
    }
