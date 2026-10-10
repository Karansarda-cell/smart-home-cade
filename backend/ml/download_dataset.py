import os
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# --- Directory Configuration ---
RAW_DIR = os.path.abspath("../data/raw")
PROCESSED_DIR = os.path.abspath("../data/processed")
DB_PATH = os.path.abspath("app/cade_audit_log.db")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# 1. UCI ML Repository (Occupancy & Environment Benchmark)
def fetch_uci_occupancy_dataset() -> bool:
    print("\n[Source 1/4] Fetching UCI Smart Room Dataset...")
    url = "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip"
    
    try:
        req = urllib.request.urlopen(url, timeout=10)
        zip_file = zipfile.ZipFile(io.BytesIO(req.read()))
        zip_file.extractall(RAW_DIR)
        
        raw_csv_path = os.path.join(RAW_DIR, "datatraining.txt")
        if os.path.exists(raw_csv_path):
            df_raw = pd.read_csv(raw_csv_path)
            
            # Map UCI columns to CADE Feature Schema
            df_cleaned = pd.DataFrame({
                "filtered_temperature": df_raw["Temperature"],
                "filtered_air_quality": df_raw["CO2"],
                "motion": df_raw["Occupancy"],
                "gas_raw": (df_raw["CO2"] * 0.85).astype(int),
                "data_source": "UCI_Benchmark"
            })
            
            output_path = os.path.join(PROCESSED_DIR, "uci_cleaned_room_data.csv")
            df_cleaned.to_csv(output_path, index=False)
            print(f" -> Success: Processed {len(df_cleaned)} rows from UCI.")
            return True
    except Exception as e:
        print(f" -> Warning: Could not fetch UCI dataset ({e})")
        return False


# 2. Kaggle Open API Ingestion
def fetch_kaggle_dataset(dataset_identifier: str = "atulanand22/smart-home-dataset-with-weather-data") -> bool:
    print(f"\n[Source 2/4] Attempting Kaggle API Download ({dataset_identifier})...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_identifier, path=RAW_DIR, unzip=True)
        print(" -> Success: Downloaded and extracted dataset from Kaggle.")
        return True
    except Exception as e:
        print(f" -> Skip: Kaggle download bypassed ({e}). Ensure ~/.kaggle/kaggle.json exists if needed.")
        return False


# 3. Local SQLite Physical Telemetry Extraction
def fetch_local_physical_telemetry() -> bool:
    print("\n[Source 3/4] Extracting Local Physical Telemetry from SQLite...")
    if not os.path.exists(DB_PATH):
        print(f" -> Skip: SQLite DB not found at {DB_PATH}. Run FastAPI backend to generate logs.")
        return False

    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        query = "SELECT * FROM audit_logs"
        df = pd.read_sql(query, con=engine)
        
        if len(df) == 0:
            print(" -> Skip: No physical log records present in database yet.")
            return False

        df_cleaned = pd.DataFrame({
            "filtered_temperature": df["filtered_temperature"],
            "filtered_air_quality": df["filtered_air_quality"],
            "motion": df["motion"].astype(int),
            "gas_raw": df["gas_raw"],
            "data_source": "Physical_ESP32_SQLite"
        })
        
        output_path = os.path.join(PROCESSED_DIR, "physical_telemetry.csv")
        df_cleaned.to_csv(output_path, index=False)
        print(f" -> Success: Extracted {len(df_cleaned)} physical records from SQLite.")
        return True
    except Exception as e:
        print(f" -> Warning: SQLite extraction error ({e})")
        return False


# 4. Fallback Synthetic Environmental Generator
def generate_synthetic_benchmark():
    print("\n[Source 4/4] Generating Baseline Synthetic Benchmark Dataset...")
    np.random.seed(42)
    n_samples = 1500

    temperature = np.random.normal(loc=25.0, scale=4.0, size=n_samples)
    air_quality = np.random.exponential(scale=80.0, size=n_samples) + 20.0
    motion = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    gas_raw = air_quality * np.random.uniform(0.8, 1.2, size=n_samples)

    df = pd.DataFrame({
        "filtered_temperature": np.round(temperature, 2),
        "filtered_air_quality": np.round(air_quality, 2),
        "motion": motion,
        "gas_raw": np.round(gas_raw, 2),
        "data_source": "Synthetic_Generator"
    })

    output_path = os.path.join(PROCESSED_DIR, "synthetic_benchmark.csv")
    df.to_csv(output_path, index=False)
    print(f" -> Success: Generated {n_samples} baseline sample rows.")


# --- Master Merger ---
def merge_all_sources_to_master():
    print("\n=== Merging All Sources into Master Dataset ===")
    processed_files = [
        os.path.join(PROCESSED_DIR, "uci_cleaned_room_data.csv"),
        os.path.join(PROCESSED_DIR, "physical_telemetry.csv"),
        os.path.join(PROCESSED_DIR, "synthetic_benchmark.csv")
    ]
    
    dataframes = []
    for filepath in processed_files:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            dataframes.append(df)
            print(f" + Included: {os.path.basename(filepath)} ({len(df)} rows)")

    if dataframes:
        master_df = pd.concat(dataframes, ignore_index=True)
        
        # Ground truth hazard label derivation for ML Training
        master_df["hazard_label"] = master_df.apply(
            lambda row: 1 if (row["gas_raw"] > 350 or row["filtered_air_quality"] > 300) else 0,
            axis=1
        )
        
        master_path = os.path.join(PROCESSED_DIR, "master_training_dataset.csv")
        master_df.to_csv(master_path, index=False)
        print(f"\n Master Dataset Ready! Total rows: {len(master_df)}")
        print(f" Saved to: {master_path}")
    else:
        print(" Error: No datasets were successfully loaded or generated.")


if __name__ == "__main__":
    print("==================================================")
    print("   CADE Multi-Source Ingestion & Fusion Pipeline  ")
    print("==================================================")
    
    fetch_uci_occupancy_dataset()
    fetch_kaggle_dataset()
    fetch_local_physical_telemetry()
    generate_synthetic_benchmark()
    
    merge_all_sources_to_master()