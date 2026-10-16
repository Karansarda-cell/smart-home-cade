from __future__ import annotations

import io
import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ML_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
APP_DIR = BACKEND_DIR / "app"
DB_PATH = APP_DIR / "cade_audit_log.db"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_uci_occupancy_dataset() -> bool:
    print("\n[Source 1] Fetching UCI Occupancy Detection dataset...")
    url = "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip"
    try:
        req = urllib.request.urlopen(url, timeout=30)
        zip_file = zipfile.ZipFile(io.BytesIO(req.read()))
        zip_file.extractall(RAW_DIR)

        candidates = list(RAW_DIR.rglob("datatraining.txt"))
        if not candidates:
            print(" -> Warning: datatraining.txt not found")
            return False

        df_raw = pd.read_csv(candidates[0])
        df_cleaned = pd.DataFrame(
            {
                "filtered_temperature": df_raw["Temperature"],
                "filtered_air_quality": df_raw["CO2"],
                "motion": df_raw["Occupancy"].astype(int),
                "gas_raw": (df_raw["CO2"] * 0.85).astype(int),
                "data_source": "UCI_Benchmark",
            }
        )
        out = PROCESSED_DIR / "uci_cleaned_room_data.csv"
        df_cleaned.to_csv(out, index=False)
        print(f" -> Success: {len(df_cleaned)} rows → {out}")
        return True
    except Exception as e:
        print(f" -> Warning: UCI fetch failed ({e})")
        return False


def generate_synthetic_benchmark(n_samples: int = 2000) -> Path:
    print("\n[Source 2] Generating synthetic classroom telemetry...")
    np.random.seed(42)
    n_normal = int(n_samples * 0.75)
    n_hazard = n_samples - n_normal

    temp_n = np.random.normal(24.0, 3.0, n_normal)
    air_n = np.random.exponential(60.0, n_normal) + 30.0
    motion_n = np.random.choice([0, 1], n_normal, p=[0.6, 0.4])
    gas_n = air_n * np.random.uniform(0.7, 1.0, n_normal)

    temp_h = np.random.normal(33.0, 4.0, n_hazard)
    air_h = np.random.uniform(280, 480, n_hazard)
    motion_h = np.random.choice([0, 1], n_hazard, p=[0.3, 0.7])
    gas_h = np.random.uniform(320, 500, n_hazard)

    df = pd.DataFrame(
        {
            "filtered_temperature": np.round(np.concatenate([temp_n, temp_h]), 2),
            "filtered_air_quality": np.round(np.concatenate([air_n, air_h]), 2),
            "motion": np.concatenate([motion_n, motion_h]).astype(int),
            "gas_raw": np.round(np.concatenate([gas_n, gas_h]), 0).astype(int),
            "data_source": "Synthetic_Generator",
        }
    )
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = PROCESSED_DIR / "synthetic_benchmark.csv"
    df.to_csv(out, index=False)
    print(f" -> Success: {len(df)} rows → {out}")
    return out


def seed_sqlite_physical_samples(n: int = 300) -> bool:
    print("\n[Source 3] Seeding SQLite with demo physical events...")
    try:
        sys.path.insert(0, str(BACKEND_DIR))
        from app.database import Base, SessionLocal, engine
        from app.models import AuditLog

        Base.metadata.create_all(bind=engine)
        np.random.seed(7)

        db = SessionLocal()
        existing = db.query(AuditLog).count()
        if existing >= n:
            print(f" -> Skip: DB already has {existing} rows")
            db.close()
            return True

        rows = []
        for i in range(n):
            is_hazard = i % 5 == 0
            raw_t = float(np.random.normal(33 if is_hazard else 24, 2))
            raw_a = float(
                np.random.uniform(320, 450)
                if is_hazard
                else np.random.uniform(40, 150)
            )
            raw_g = int(
                np.random.uniform(360, 480)
                if is_hazard
                else np.random.uniform(50, 200)
            )
            motion = bool(np.random.choice([0, 1]))
            fan = 100.0 if is_hazard else float(np.clip((raw_t - 20) * 5, 0, 80))
            hazard_lvl = 85.0 if is_hazard else float(np.random.uniform(5, 40))
            alert = "EMERGENCY_GAS_HAZARD" if is_hazard else "NORMAL"

            rows.append(
                AuditLog(
                    raw_temperature=raw_t,
                    raw_air_quality=raw_a,
                    raw_motion=motion,
                    raw_gas=raw_g,
                    raw_payload=None,
                    filtered_temperature=raw_t,
                    filtered_air_quality=raw_a,
                    motion=motion,
                    gas_raw=raw_g,
                    temp_anomaly=False,
                    air_anomaly=False,
                    fuzzy_fan_speed=fan,
                    fuzzy_hazard_level=hazard_lvl,
                    relay_fan_pwm=fan,
                    buzzer=is_hazard,
                    door_lock="UNLOCKED",
                    alert_status=alert,
                    system_mode="DISARMED",
                    ml_label=None,
                    ml_confidence=None,
                )
            )

        db.add_all(rows)
        db.commit()
        total = db.query(AuditLog).count()
        db.close()
        print(f" -> Success: DB has {total} rows at {DB_PATH}")
        return True
    except Exception as e:
        print(f" -> Warning: SQLite seed failed ({e})")
        return False


def export_physical_from_db() -> bool:
    print("\n[Source 3b] Exporting physical telemetry from SQLite...")
    try:
        from sqlalchemy import create_engine

        if not DB_PATH.exists():
            print(" -> No DB file yet")
            return False

        engine = create_engine(f"sqlite:///{DB_PATH}")
        df = pd.read_sql("SELECT * FROM audit_logs", con=engine)
        if df.empty:
            print(" -> DB empty")
            return False

        out = pd.DataFrame(
            {
                "filtered_temperature": df["filtered_temperature"],
                "filtered_air_quality": df["filtered_air_quality"],
                "motion": df["motion"].astype(int),
                "gas_raw": df["gas_raw"].astype(int),
                "data_source": "Physical_SQLite",
            }
        )
        path = PROCESSED_DIR / "physical_telemetry.csv"
        out.to_csv(path, index=False)
        print(f" -> Success: {len(out)} rows → {path}")
        return True
    except Exception as e:
        print(f" -> Warning: export failed ({e})")
        return False


def merge_all_sources_to_master():
    print("\n[Merge] Building master_training_dataset.csv...")
    files = [
        PROCESSED_DIR / "uci_cleaned_room_data.csv",
        PROCESSED_DIR / "physical_telemetry.csv",
        PROCESSED_DIR / "synthetic_benchmark.csv",
    ]
    frames = []
    for fp in files:
        if fp.exists():
            part = pd.read_csv(fp)
            frames.append(part)
            print(f" + {fp.name}: {len(part)} rows")

    if not frames:
        print(" Error: no source CSVs found")
        return None

    master = pd.concat(frames, ignore_index=True)
    master["hazard_label"] = master.apply(
        lambda r: 1
        if (float(r["gas_raw"]) > 350 or float(r["filtered_air_quality"]) > 300)
        else 0,
        axis=1,
    )
    master_path = PROCESSED_DIR / "master_training_dataset.csv"
    master.to_csv(master_path, index=False)
    print(f"\n Master ready: {len(master)} rows → {master_path}")
    print(master["hazard_label"].value_counts().to_string())
    return master_path


if __name__ == "__main__":
    print("=" * 50)
    print("  CADE Data Ingestion Pipeline")
    print("=" * 50)
    fetch_uci_occupancy_dataset()
    generate_synthetic_benchmark()
    seed_sqlite_physical_samples(300)
    export_physical_from_db()
    merge_all_sources_to_master()
    print("\nDone. Next: python backend/ml/train_model.py")