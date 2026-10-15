# smart-home-cade
Architected a Fedora-hosted FastAPI and MQTT Context-Aware Decision Engine incorporating Mamdani Fuzzy Logic and Random Forest ML to automate real-time smart classroom environmental controls and safety overrides.
smart-home-cade/
├── README.md
├── requirements.txt          ← UPDATE
└── backend/
    ├── app/
    │   ├── __init__.py         ← ADD
    │   ├── anomaly_filter.py   ← RENAME (was anomly_filter.py)
    │   ├── config.py           ← keep
    │   ├── conflict_resolver.py← keep (optional later tweak)
    │   ├── database.py         ← ADD
    │   ├── fuzzy_engine.py     ← keep (optional later tweak)
    │   ├── main.py             ← REPLACE
    │   ├── ml_model.py         ← REPLACE (was empty)
    │   ├── models.py           ← ADD
    │   └── mqtt_client.py      ← keep (imports anomaly_filter — OK after rename)
    └── ml/
        ├── download_dataset.py
        ├── export_db_data.py
        └── train_model.py      ← optional path/message fix later

