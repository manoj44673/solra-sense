
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

try:
    from ml.data_loader import build_dataset
except ImportError:
    from data_loader import build_dataset

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL = ["PLANT_ID", "SOURCE_KEY"]
NUMERIC_FAULT = [
    "DC_POWER", "DAILY_YIELD", "IRRADIATION", "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE", "HOUR", "MINUTE", "DAY_OF_WEEK", "MONTH", "DAY_OF_YEAR"
]
NUMERIC_GEN = [
    "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
    "HOUR", "MINUTE", "DAY_OF_WEEK", "MONTH", "DAY_OF_YEAR"
]

def make_preprocessor(numeric_cols):
    return ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
    ])

def train_models(df=None):
    if df is None:
        df, _ = build_dataset(save=True)

    # Fault classifier: balanced sample, with AC_POWER excluded from features
    # to prevent directly leaking the rule used to create FAULT_FLAG.
    fault_pos = df[df["FAULT_FLAG"] == 1]
    fault_neg = df[df["FAULT_FLAG"] == 0].sample(
        n=min(45000, int((df["FAULT_FLAG"] == 0).sum())), random_state=42
    )
    fault_df = pd.concat([fault_pos, fault_neg], ignore_index=True).sample(frac=1, random_state=42)

    Xf = fault_df[NUMERIC_FAULT + CATEGORICAL]
    yf = fault_df["FAULT_FLAG"].astype(int)
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        Xf, yf, test_size=0.20, random_state=42, stratify=yf
    )
    fault_model = Pipeline([
        ("preprocessor", make_preprocessor(NUMERIC_FAULT)),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=18, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42
        ))
    ])
    fault_model.fit(Xf_train, yf_train)
    pf = fault_model.predict(Xf_test)
    fault_metrics = {
        "accuracy": float(accuracy_score(yf_test, pf)),
        "precision": float(precision_score(yf_test, pf, zero_division=0)),
        "recall": float(recall_score(yf_test, pf, zero_division=0)),
        "f1": float(f1_score(yf_test, pf, zero_division=0)),
        "confusion_matrix": confusion_matrix(yf_test, pf).tolist(),
        "test_size": int(len(yf_test)),
        "fault_rate": float(df["FAULT_FLAG"].mean()),
    }

    # Generation regressor
    gen_df = df.sample(n=min(60000, len(df)), random_state=42)
    Xg = gen_df[NUMERIC_GEN + CATEGORICAL]
    yg = gen_df["AC_POWER"].astype(float)
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        Xg, yg, test_size=0.20, random_state=42
    )
    gen_model = Pipeline([
        ("preprocessor", make_preprocessor(NUMERIC_GEN)),
        ("regressor", RandomForestRegressor(
            n_estimators=50, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        ))
    ])
    gen_model.fit(Xg_train, yg_train)
    pg = gen_model.predict(Xg_test)
    gen_metrics = {
        "mae": float(mean_absolute_error(yg_test, pg)),
        "mse": float(mean_squared_error(yg_test, pg)),
        "rmse": float(np.sqrt(mean_squared_error(yg_test, pg))),
        "r2": float(r2_score(yg_test, pg)),
        "test_size": int(len(yg_test)),
    }

    joblib.dump(fault_model, MODEL_DIR / "fault_model.joblib")
    joblib.dump(gen_model, MODEL_DIR / "generation_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"fault": fault_metrics, "generation": gen_metrics}, f, indent=2)
    return fault_metrics, gen_metrics

def load_models():
    try:
        return (
            joblib.load(MODEL_DIR / "fault_model.joblib"),
            joblib.load(MODEL_DIR / "generation_model.joblib"),
        )
    except Exception:
        # Rebuild stale/incompatible serialized models automatically.
        df, _ = build_dataset(save=True)
        train_models(df)
        return (
            joblib.load(MODEL_DIR / "fault_model.joblib"),
            joblib.load(MODEL_DIR / "generation_model.joblib"),
        )

if __name__ == "__main__":
    df, _ = build_dataset(save=True)
    f, g = train_models(df)
    print(json.dumps({"fault": f, "generation": g}, indent=2))
