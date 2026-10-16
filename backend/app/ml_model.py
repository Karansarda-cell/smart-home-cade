from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "cade_ml_model.pkl"

FEATURE_COLUMNS = [
    "filtered_temperature",
    "filtered_air_quality",
    "motion",
    "gas_raw",
]


class HazardMLModel:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if not self.model_path.exists():
            print(
                f"[ML] Model not found at {self.model_path}. "
                "Run: python backend/ml/train_model.py"
            )
            self._model = None
            return False
        try:
            self._model = joblib.load(self.model_path)
            print(f"[ML] Loaded model from {self.model_path}")
            return True
        except Exception as e:
            print(f"[ML] Load failed: {e}")
            self._model = None
            return False

    def predict(
        self,
        temperature: float,
        air_quality: float,
        motion: bool,
        gas_raw: int | float,
    ) -> dict:
        if self._model is None:
            raise RuntimeError("ML model is not loaded.")

        input_data = pd.DataFrame(
            [
                {
                    "filtered_temperature": float(temperature),
                    "filtered_air_quality": float(air_quality),
                    "motion": 1 if motion else 0,
                    "gas_raw": float(gas_raw),
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        code = int(self._model.predict(input_data)[0])
        probs: list[float] = []
        if hasattr(self._model, "predict_proba"):
            probs = [float(x) for x in self._model.predict_proba(input_data)[0].tolist()]

        confidence = max(probs) if probs else None
        return {
            "prediction_label": "HAZARD_ALERT" if code == 1 else "NORMAL",
            "prediction_code": code,
            "confidence_scores": probs,
            "confidence": confidence,
        }


hazard_model = HazardMLModel()