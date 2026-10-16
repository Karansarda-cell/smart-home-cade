from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.ml_model import hazard_model
from app.models import AuditLog
from app.mqtt_client import MQTTHandler

mqtt_service = MQTTHandler()
ALLOWED_MODES = {"ARMED", "DISARMED"}


class ModeUpdate(BaseModel):
    mode: str

    @field_validator("mode")
    @classmethod
    def mode_must_be_known(cls, v: str) -> str:
        m = v.strip().upper()
        if m not in ALLOWED_MODES:
            raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)}")
        return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    hazard_model.load()
    mqtt_service.start()
    yield
    mqtt_service.client.loop_stop()


app = FastAPI(title="Context-Aware Decision Engine (CADE)", lifespan=lifespan)


@app.get("/")
def get_status():
    return {
        "status": "CADE Backend Active",
        "system_mode": mqtt_service.system_mode,
        "ml_model_loaded": hazard_model.is_loaded,
    }


@app.post("/api/mode")
def update_mode(body: ModeUpdate):
    mqtt_service.system_mode = body.mode
    return {"status": "Mode updated", "current_mode": mqtt_service.system_mode}


@app.get("/api/logs")
def get_audit_logs(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return (
        db.query(AuditLog)
        .order_by(AuditLog.id.desc())
        .limit(limit)
        .all()
    )


@app.get("/api/metrics/summary")
def metrics_summary(
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = db.query(AuditLog).filter(AuditLog.timestamp >= since)

    total = q.count()
    emergencies = q.filter(AuditLog.alert_status == "EMERGENCY_GAS_HAZARD").count()
    intrusions = q.filter(AuditLog.alert_status == "SECURITY_INTRUSION").count()
    temp_anoms = q.filter(AuditLog.temp_anomaly.is_(True)).count()
    air_anoms = q.filter(AuditLog.air_anomaly.is_(True)).count()

    avg_temp = (
        db.query(func.avg(AuditLog.filtered_temperature))
        .filter(AuditLog.timestamp >= since)
        .scalar()
    )
    avg_air = (
        db.query(func.avg(AuditLog.filtered_air_quality))
        .filter(AuditLog.timestamp >= since)
        .scalar()
    )

    return {
        "window_hours": hours,
        "total_events": total,
        "emergencies": emergencies,
        "intrusions": intrusions,
        "temp_anomalies": temp_anoms,
        "air_anomalies": air_anoms,
        "avg_filtered_temperature": round(float(avg_temp), 2) if avg_temp is not None else None,
        "avg_filtered_air_quality": round(float(avg_air), 2) if avg_air is not None else None,
    }


@app.get("/api/metrics/timeseries")
def metrics_timeseries(
    metric: str = Query("filtered_temperature"),
    hours: int = Query(6, ge=1, le=72),
    limit: int = Query(500, ge=10, le=2000),
    db: Session = Depends(get_db),
):
    allowed = {
        "filtered_temperature",
        "filtered_air_quality",
        "raw_temperature",
        "raw_air_quality",
        "relay_fan_pwm",
        "fuzzy_hazard_level",
        "gas_raw",
    }
    if metric not in allowed:
        raise HTTPException(400, detail=f"metric must be one of {sorted(allowed)}")

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    col = getattr(AuditLog, metric)
    rows = (
        db.query(AuditLog.timestamp, col)
        .filter(AuditLog.timestamp >= since)
        .order_by(AuditLog.id.asc())
        .limit(limit)
        .all()
    )
    return {
        "metric": metric,
        "points": [
            {
                "timestamp": ts.isoformat() if ts else None,
                "value": float(val) if val is not None else None,
            }
            for ts, val in rows
        ],
    }


@app.post("/api/predict")
def predict_hazard(
    temperature: float,
    air_quality: float,
    motion: bool,
    gas_raw: int,
):
    if not hazard_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run: python backend/ml/train_model.py",
        )
    try:
        return hazard_model.predict(
            temperature=temperature,
            air_quality=air_quality,
            motion=motion,
            gas_raw=gas_raw,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e