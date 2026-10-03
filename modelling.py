"""
==============================================================
 modelling.py
 Membangun model klasifikasi churn dengan MLflow Tracking UI
 (versi Basic: MLflow autolog, tracking lokal)
==============================================================

Dataset yang dipakai adalah dataset hasil preprocessing dari
repository `Eksperimen_SML_Bayu-Septiawan` (BUKAN data raw).

Cara pakai (dari folder Membangun_model/):
    python modelling.py
    mlflow ui --port 5000     # buka http://localhost:5000
"""

import os

# MLflow 3.x: izinkan file store lokal ./mlruns (mode lama) agar kompatibel
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------- Konfigurasi ----------------
DATA_PATH = os.getenv("DATA_PATH", "namadataset_preprocessing/telco_churn_preprocessing.csv")
EXPERIMENT_NAME = "Telco_Churn_Modelling"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str):
    """Memuat dataset siap latih lalu membaginya (train/test, stratified)."""
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def main() -> None:
    # Tracking lokal (folder ./mlruns) sesuai kriteria Basic
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    print(f"Data latih : {X_train.shape} | Data uji: {X_test.shape}")

    # --- Basic: menggunakan MLflow autolog ---
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="logistic-regression-autolog") as run:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print("\n=== Evaluasi model baseline ===")
        print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
        print("Precision:", round(precision_score(y_test, y_pred), 4))
        print("Recall   :", round(recall_score(y_test, y_pred), 4))
        print("F1-score :", round(f1_score(y_test, y_pred), 4))
        print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 4))

        print("\nRun ID   :", run.info.run_id)
        print("Model URI: runs:/" + run.info.run_id + "/model")
        print("\nLihat tracking UI : mlflow ui --port 5000")


if __name__ == "__main__":
    main()
