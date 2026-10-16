from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

ML_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ML_DIR.parent
DB_PATH = BACKEND_DIR / "app" / "cade_audit_log.db"
OUT_PATH = BACKEND_DIR / "data" / "processed" / "physical_telemetry.csv"


def export_physical_data() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No database at {DB_PATH}")

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM audit_logs", con=engine)
    print(f"[Export] {len(df)} rows from SQLite")

    features = pd.DataFrame(
        {
            "filtered_temperature": df["filtered_temperature"],
            "filtered_air_quality": df["filtered_air_quality"],
            "motion": df["motion"].astype(int),
            "gas_raw": df["gas_raw"].astype(int),
            "data_source": "Physical_SQLite",
        }
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_PATH, index=False)
    print(f"[Export] Saved → {OUT_PATH}")


if __name__ == "__main__":
    export_physical_data()