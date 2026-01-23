import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.age_median_ = float(pd.to_numeric(X["Age"], errors="coerce").median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Fare_log
        X["Fare"] = pd.to_numeric(X["Fare"], errors="coerce")
        X["Fare_log"] = np.log1p(X["Fare"])

        # Has_Cabin_Number
        cabin = X["Cabin"].astype("object")
        X["Has_Cabin_Number"] = (cabin.notna() & (cabin.astype(str).str.strip() != "")).astype(int)

        # Title_transformed (неизвестное -> Rare)
        allowed = {"Master", "Miss", "Mr", "Mrs"}
        title = X["Title"].astype(str).str.strip()
        X["Title_transformed"] = title.where(title.isin(list(allowed)), "Rare")

        # FamilySize_cluster
        X["SibSp"] = pd.to_numeric(X["SibSp"], errors="coerce").fillna(0).astype(int)
        X["Parch"] = pd.to_numeric(X["Parch"], errors="coerce").fillna(0).astype(int)
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1

        X["FamilySize_cluster"] = pd.cut(
            X["FamilySize"],
            bins=[0, 1, 4, np.inf],
            labels=["Alone", "MidSizeFamily", "LargeFamily"],
            include_lowest=True
        ).astype(str)

        # AgeGroup_median (как ты сказал)
        age_filled = pd.to_numeric(X["Age"], errors="coerce").fillna(self.age_median_)
        X["AgeGroup_median"] = pd.cut(
            age_filled,
            bins=[0, 12, 20, 40, 60, np.inf],
            labels=["Child", "Teenager", "Adult", "MiddleAged", "Senior"],
            include_lowest=True
        ).astype(str)

        return X


def build_custom_pipeline(best_params: dict, random_state: int = 42) -> Pipeline:
    numeric_features = ["Fare_log", "Pclass", "Has_Cabin_Number"]
    categorical_features = ["Sex", "AgeGroup_median", "FamilySize_cluster", "Title_transformed"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop"
    )

    # базовые обязательные штуки + best_params поверх
    model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        **best_params
    )

    return Pipeline([
        ("fe", TitanicFeatureEngineer()),
        ("pre", pre),
        ("model", model),
    ])
