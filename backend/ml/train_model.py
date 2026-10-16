from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ML_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ML_DIR.parent
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
MASTER_PATH = PROCESSED_DIR / "master_training_dataset.csv"
MODEL_PATH = BACKEND_DIR / "app" / "cade_ml_model.pkl"

FEATURE_COLS = [
    "filtered_temperature",
    "filtered_air_quality",
    "motion",
    "gas_raw",
]


def train_decision_engine_model() -> None:
    print("=" * 50)
    print("  CADE ML Training (Random Forest)")
    print("=" * 50)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset missing: {MASTER_PATH}\n"
            "Run: python backend/ml/download_dataset.py first"
        )

    print(f"\n[1/5] Loading {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)
    print(f" -> samples: {len(df)}")
    if "data_source" in df.columns:
        print(df["data_source"].value_counts().to_string())

    if "hazard_label" not in df.columns:
        df["hazard_label"] = df.apply(
            lambda r: 1
            if (float(r["gas_raw"]) > 350 or float(r["filtered_air_quality"]) > 300)
            else 0,
            axis=1,
        )

    X = df[FEATURE_COLS]
    y = df["hazard_label"].astype(int)

    print(f"\n[2/5] Features: {FEATURE_COLS}")
    print(y.value_counts().to_string())

    print("\n[3/5] Train/test split 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n[4/5] Training...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n================ EVALUATION ================")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "HAZARD_ALERT"]))
    print(confusion_matrix(y_test, y_pred))
    print("============================================")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[5/5] Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_decision_engine_model()