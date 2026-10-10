import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Directory & File Paths ---
PROCESSED_DIR = os.path.abspath("../data/processed")
MASTER_DATASET_PATH = os.path.join(PROCESSED_DIR, "master_training_dataset.csv")
MODEL_OUTPUT_PATH = os.path.abspath("app/cade_ml_model.pkl")


def train_decision_engine_model():
    print("==================================================")
    print("   CADE ML Engine Training Pipeline (Random Forest)")
    print("==================================================")

    # 1. Check if Master Dataset exists
    if not os.path.exists(MASTER_DATASET_PATH):
        raise FileNotFoundError(
            f"Master dataset missing at '{MASTER_DATASET_PATH}'. "
            "Run 'python ml/download_external_datasets.py' first!"
        )

    # 2. Load Master Dataset
    print(f"\n[1/5] Loading Master Dataset from: {MASTER_DATASET_PATH}")
    df = pd.read_csv(MASTER_DATASET_PATH)
    print(f" -> Total dataset samples: {len(df)}")
    
    if "data_source" in df.columns:
        print(" -> Data breakdown by source:")
        print(df["data_source"].value_counts().to_string())

    # 3. Feature Matrix (X) & Target Label (y)
    feature_cols = ["filtered_temperature", "filtered_air_quality", "motion", "gas_raw"]
    
    # Ensure ground-truth label exists
    if "hazard_label" not in df.columns:
        print(" -> Deriving 'hazard_label' based on threshold safety rules...")
        df["hazard_label"] = df.apply(
            lambda row: 1 if (row["gas_raw"] > 350 or row["filtered_air_quality"] > 300) else 0,
            axis=1
        )

    X = df[feature_cols]
    y = df["hazard_label"]

    print(f"\n[2/5] Extracted Features: {feature_cols}")
    print(f" -> Target distribution (0: Normal, 1: Hazard Alert):")
    print(y.value_counts().to_string())

    # 4. Train/Test Stratified Split
    print("\n[3/5] Splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Model Initialization & Training
    print("\n[4/5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n================ EVALUATION METRICS ================")
    print(f"Model Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "HAZARD_ALERT"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("====================================================")

    # 7. Model Serialization & Export
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\n[5/5] Success! Trained model serialized to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_decision_engine_model()