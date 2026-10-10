import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from app.mqtt_client import MQTTHandler
from app.database import get_db, engine, Base
from app.models import AuditLog

# Global handle for ML model
ml_model = None
mqtt_service = MQTTHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    
    # 1. Initialize SQLite Database Tables
    Base.metadata.create_all(bind=engine)
    
    # 2. Load ML Model (.pkl) if available
    model_path = os.path.abspath("app/cade_ml_model.pkl")
    if os.path.exists(model_path):
        try:
            ml_model = joblib.load(model_path)
            print(f"[ML Pipeline] Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"[ML Pipeline Error] Could not load model: {e}")
    else:
        print("[ML Pipeline Warning] cade_ml_model.pkl not found. Run 'python ml/train_model.py' first.")

    # 3. Start MQTT Loop
    mqtt_service.start()
    
    yield
    
    # Clean shutdown
    mqtt_service.client.loop_stop()

app = FastAPI(title="Context-Aware Decision Engine (CADE)", lifespan=lifespan)

# --- Status & Mode Endpoints ---

@app.get("/")
def get_status():
    return {
        "status": "CADE Backend Active on Fedora",
        "system_mode": mqtt_service.system_mode,
        "ml_model_loaded": ml_model is not None
    }

@app.post("/api/mode")
def update_mode(mode: str):
    mqtt_service.system_mode = mode.upper()
    return {"status": "Mode updated", "current_mode": mqtt_service.system_mode}

# --- Database Audit Logs Endpoint ---

@app.get("/api/logs")
def get_audit_logs(limit: int = 50, db: Session = Depends(get_db)):
    """
    Returns the latest audit logs from SQLite (cade_audit_log.db).
    """
    logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
    return logs

# --- Real-Time ML Inference Endpoint ---

@app.post("/api/predict")
def predict_hazard(temperature: float, air_quality: float, motion: bool, gas_raw: int):
    """
    Evaluates real-time sensor telemetry using the trained Random Forest model.
    """
    if ml_model is None:
        raise HTTPException(
            status_code=503, 
            detail="ML Model not loaded. Ensure cade_ml_model.pkl is generated."
        )

    # Feature matrix matching the training schema in train_model.py
    input_data = pd.DataFrame([{
        "filtered_temperature": temperature,
        "filtered_air_quality": air_quality,
        "motion": 1 if motion else 0,
        "gas_raw": gas_raw
    }])

    prediction = ml_model.predict(input_data)[0]
    probabilities = ml_model.predict_proba(input_data)[0].tolist() if hasattr(ml_model, "predict_proba") else []

    return {
        "prediction_label": "HAZARD_ALERT" if prediction == 1 else "NORMAL",
        "prediction_code": int(prediction),
        "confidence_scores": probabilities
    }