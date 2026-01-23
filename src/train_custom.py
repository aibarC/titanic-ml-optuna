import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from custom_pipeline import build_custom_pipeline

DATA_PATH = Path("data") / "raw" / "train_titanic.csv"
PARAMS_PATH = Path("artifacts") / "model_data" / "best_params.json"
OUT_PATH = Path("artifacts") / "model_data" / "xgb_pipeline_raw.joblib"


def load_best_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    # иногда json хранит числа как строки — приводим минимум
    int_keys = {"max_depth", "n_estimators", "min_child_weight"}
    float_keys = {"learning_rate", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"}

    for k in list(params.keys()):
        if k in int_keys:
            params[k] = int(params[k])
        if k in float_keys:
            params[k] = float(params[k])

    return params


def make_raw_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Sex", "Age", "Fare", "Pclass", "SibSp", "Parch", "Cabin"]].copy()

    # Title берём из колонки Title если она есть, иначе один раз извлекаем из Name
    if "Title" in df.columns:
        X["Title"] = df["Title"]
    else:
        titles = df["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False)
        titles = titles.fillna("Rare").str.strip()
        allowed = {"Master", "Miss", "Mr", "Mrs"}
        X["Title"] = titles.where(titles.isin(list(allowed)), "Rare")

    return X


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["Survived"].astype(int).values
    X = make_raw_X(df)

    best_params = load_best_params(PARAMS_PATH)
    pipe = build_custom_pipeline(best_params=best_params, random_state=42)

    pipe.fit(X, y)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_PATH)
    print("Saved tuned pipeline:", OUT_PATH)


if __name__ == "__main__":
    main()
