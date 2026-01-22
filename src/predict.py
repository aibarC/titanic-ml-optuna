# src/predict.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd


# ---------- Paths (по умолчанию под твою структуру) ----------
DEFAULT_ARTIFACTS_DIR = Path("artifacts") / "model_data"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "xgb_pipeline.joblib"
DEFAULT_FEATURES_PATH = Path("artifacts") / "features.json"
DEFAULT_THRESHOLD_PATH = DEFAULT_ARTIFACTS_DIR / "threshold.json"


# ---------- Helpers ----------
def _load_json(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_features(obj: Any) -> List[str]:
    """
    Поддерживает разные форматы:
    - ["Age","Sex",...]
    - {"features":[...]}
    - {"feature_names":[...]}
    - {"columns":[...]}
    """
    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            return obj
        raise ValueError("features.json: ожидается список строк (feature names).")

    if isinstance(obj, dict):
        for key in ("features", "feature_names", "columns", "input_features"):
            if key in obj and isinstance(obj[key], list):
                if all(isinstance(x, str) for x in obj[key]):
                    return obj[key]
        # иногда сохраняют как {"0":"Age","1":"Fare"...}
        if all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, str) for v in obj.values()):
            # сортируем по ключам, если ключи - числа в строке
            try:
                items = sorted(obj.items(), key=lambda kv: int(kv[0]))
                return [v for _, v in items]
            except Exception:
                return list(obj.values())

    raise ValueError("Не смог понять формат features.json.")


def _extract_threshold(obj: Any, default: float = 0.5) -> float:
    """
    Поддерживает:
    - число: 0.42
    - {"threshold":0.42}
    - {"best_threshold":0.42}
    - {"f1_threshold":0.42}
    """
    if obj is None:
        return float(default)

    if isinstance(obj, (int, float)):
        return float(obj)

    if isinstance(obj, dict):
        for key in ("threshold", "best_threshold", "f1_threshold", "optimal_threshold"):
            if key in obj and isinstance(obj[key], (int, float)):
                return float(obj[key])

    return float(default)


def _coerce_df_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Лёгкая попытка привести типы:
    - числовые строки → float
    - остальное оставляем как есть (для OneHot/Ordinal в pipeline)
    """
    out = df.copy()
    for col in out.columns:
        # пробуем конвертнуть в число, если возможно
        out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


# ---------- Core ----------
@dataclass
class TitanicPredictor:
    model: Any
    features: List[str]
    threshold: float = 0.5

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        features_path: Union[str, Path] = DEFAULT_FEATURES_PATH,
        threshold_path: Union[str, Path] = DEFAULT_THRESHOLD_PATH,
        default_threshold: float = 0.5,
    ) -> "TitanicPredictor":
        model_path = Path(model_path)
        features_path = Path(features_path)
        threshold_path = Path(threshold_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        model = joblib.load(model_path)

        features_obj = _load_json(features_path)
        features = _extract_features(features_obj)

        thr = default_threshold
        if threshold_path.exists():
            thr_obj = _load_json(threshold_path)
            thr = _extract_threshold(thr_obj, default=default_threshold)

        return cls(model=model, features=features, threshold=thr)

    def build_input(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Собирает DataFrame строго по self.features.
        Если каких-то фич нет — ставит NaN (pipeline должен уметь это обработать).
        """
        row: Dict[str, Any] = {}
        for f in self.features:
            row[f] = user_input.get(f, np.nan)

        df = pd.DataFrame([row], columns=self.features)
        df = _coerce_df_types(df)
        return df

    def predict_proba(self, user_input: Dict[str, Any]) -> float:
        X = self.build_input(user_input)

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Загруженная модель/pipeline не имеет метода predict_proba().")

        proba = self.model.predict_proba(X)
        # обычно shape (n,2), берем вероятность класса 1
        proba_1 = float(proba[0, 1]) if proba.ndim == 2 and proba.shape[1] >= 2 else float(proba[0])
        return proba_1

    def predict(self, user_input: Dict[str, Any], threshold: Optional[float] = None) -> Tuple[int, float, float]:
        """
        Возвращает (pred_label, proba, used_threshold)
        """
        used_thr = float(self.threshold if threshold is None else threshold)
        p = self.predict_proba(user_input)
        pred = int(p >= used_thr)
        return pred, p, used_thr


# ---------- Convenience function (удобно для Streamlit) ----------
_predictor_singleton: Optional[TitanicPredictor] = None


def get_predictor() -> TitanicPredictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = TitanicPredictor.load()
    return _predictor_singleton


if __name__ == "__main__":
    # Мини-тест: поменяй поля под свои features.json
    predictor = TitanicPredictor.load()

    print("Loaded features count:", len(predictor.features))
    print("Threshold:", predictor.threshold)
    print("First 10 features:", predictor.features[:10])

    # Пример user_input — ВАЖНО: ключи должны совпадать с названиями фич из features.json
    example_input = {f: np.nan for f in predictor.features}
    # Попробуй заполнить пару популярных:
    for k, v in {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "Fare": 7.25,
        "SibSp": 1,
        "Parch": 0,
        "Embarked": "S",
    }.items():
        if k in example_input:
            example_input[k] = v

    pred, proba, thr = predictor.predict(example_input)
    print("Prediction:", pred, "Proba:", proba, "Threshold:", thr)
