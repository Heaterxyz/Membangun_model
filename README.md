# Membangun_model — MLflow Tracking

Pelatihan model klasifikasi **Telco Customer Churn** menggunakan MLflow Tracking UI dengan dataset hasil preprocessing (bukan raw).

## Struktur

```
Membangun_model/
├── modelling.py                     # Basic: MLflow autolog, tracking lokal (./mlruns)
├── modelling_tuning.py              # Skilled/Advanced: tuning + manual logging + DagsHub
├── namadataset_preprocessing/
│   └── telco_churn_preprocessing.csv
├── screenshoot_dashboard.jpg        # screenshot MLflow Tracking UI
├── screenshoot_artifak.jpg          # screenshot artefak pada run
├── requirements.txt
└── DagsHub.txt                      # tautan DagsHub (Advanced)
```

## Menjalankan

```bash
pip install -r requirements.txt

# Basic - autolog, tracking lokal
python modelling.py
mlflow ui --port 5000        # http://localhost:5000

# Skilled/Advanced - hyperparameter tuning + manual logging
python modelling_tuning.py   # tracking lokal ./mlruns

# Advanced - tracking online DagsHub
export DAGSHUB_USERNAME=<username>
export DAGSHUB_REPO_NAME=<repo>
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<token>
python modelling_tuning.py
```

## Ketentuan per Level

| Level | File | Yang diterapkan |
|---|---|---|
| Basic | `modelling.py` | `mlflow.sklearn.autolog()`, tracking lokal, screenshot dashboard & artefak |
| Skilled | `modelling_tuning.py` | `RandomizedSearchCV` + **manual logging** (params, metrics, model) |
| Advanced | `modelling_tuning.py` | Tracking **online DagsHub** + 3 artefak tambahan di luar autolog: `confusion_matrix.png`, `feature_importance.png`, `classification_report.txt` |
