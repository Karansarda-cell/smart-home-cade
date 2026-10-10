import pandas as pd
from sqlalchemy import create_engine
import os

# Connect to local SQLite audit DB
DB_PATH = os.path.abspath("../app/cade_audit_log.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

def export_physical_data():
    query = "SELECT * FROM audit_logs"
    df = pd.read_sql(query, con=engine)
    
    print(f"[Data Pipeline] Extracted {len(df)} rows from SQLite DB.")
    
    # Feature Selection for ML
    features = df[[
        "filtered_temperature", 
        "filtered_air_quality", 
        "motion", 
        "gas_raw", 
        "fuzzy_fan_speed", 
        "fuzzy_hazard_level", 
        "alert_status"
    ]]
    
    output_path = "../data/processed/physical_telemetry.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"[Data Pipeline] Saved feature matrix to {output_path}")

if __name__ == "__main__":
    export_physical_data()