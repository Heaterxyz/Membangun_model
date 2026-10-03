"""
==============================================================
 modelling_tuning.py
 Hyperparameter tuning + manual logging MLflow (DagsHub / lokal)
==============================================================

Checklist kriteria (dijalankan pada FILE INI, bukan modelling.py):

 [x] Skilled - Hyperparameter tuning dengan RandomizedSearchCV
 [x] Skilled - MANUAL logging (bukan autolog) dengan metrik setara autolog:
        - log_params  : seluruh parameter model & proses tuning
        - log_metrics : accuracy, precision, recall, f1, roc_auc, dst.
        - log_model   : model terbaik (signature + input_example)
 [x] Advanced - Tracking ONLINE via DagsHub (jika env diatur),
        fallback ke tracking lokal jika env kosong
 [x] Advanced - >= 2 artefak tambahan DI LUAR yang dicover autolog:
        1. artifacts/confusion_matrix.png
        2. artifacts/feature_importance.png
        3. artifacts/classification_report.txt

Cara pakai (dari folder Membangun_model/):
    # Tracking lokal:
    python modelling_tuning.py

    # Tracking online DagsHub (kriteria Advanced):
    export DAGSHUB_USERNAME=<username-dagshub>
    export DAGSHUB_REPO_NAME=<nama-repo-dagshub>
    export MLFLOW_TRACKING_USERNAME=<username-dagshub>
    export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
    python modelling_tuning.py

Lihat tracking UI:
    - Lokal  : mlflow ui --port 5000 -> http://localhost:5000
    - DagsHub: https://dagshub.com/<username-dagshub>/<nama-repo-dagshub>.mlflow
"""

import os
import shutil
import time

# MLflow 3.x: izinkan file store lokal ./mlruns (mode lama) agar kompatibel
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# ---------------- Konfigurasi ----------------
DATA_PATH = os.getenv("DATA_PATH", "namadataset_preprocessing/telco_churn_preprocessing.csv")
EXPERIMENT_NAME = "Telco_Churn_Tuning"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ruang hyperparameter yang dicari RandomizedSearchCV
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 401),
    "max_depth": [None, 6, 8, 10, 12, 16],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}
N_ITER = 12          # jumlah kombinasi yang diacak
CV_FOLDS = 3         # jumlah lipatan cross-validation
SCORING = "f1"       # metrik pemilihan model terbaik
TMP_DIR = "tmp_artifacts"


def setup_mlflow_tracking() -> str:
    """Kriteria Advanced: tracking ONLINE ke DagsHub jika env tersedia, selain itu lokal."""
    username = os.getenv("DAGSHUB_USERNAME")
    repo = os.getenv("DAGSHUB_REPO_NAME")
    if username and repo:
        try:
            import dagshub

            dagshub.init(repo_owner=username, repo_name=repo, mlflow=True)
            print(f"[tracking] DagsHub online -> {username}/{repo}")
            return "dagshub"
        except Exception as exc:  # noqa: BLE001
            print(f"[tracking] DagsHub gagal ({exc}); fallback ke lokal.")
    mlflow.set_tracking_uri("mlruns")
    print("[tracking] Lokal -> ./mlruns")
    return "local"


def load_data(path: str):
    """Memuat dataset siap latih lalu membaginya (train/test, stratified)."""
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ---------------- Pembuatan artefak tambahan ----------------
def save_confusion_matrix(y_true, y_pred, path: str) -> None:
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=["No Churn", "Churn"]
    )
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix - Model Terbaik")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_feature_importance(model, feature_names, path: str) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    importances.tail(15).plot(kind="barh", color="#4C72B0")
    plt.title("Feature Importance (Top 15) - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_classification_report(y_true, y_pred, path: str) -> None:
    report = classification_report(
        y_true, y_pred, target_names=["No Churn", "Churn"], digits=4
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("Classification Report - Random Forest (hasil tuning)\n")
        f.write("=" * 60 + "\n")
        f.write(report)


def main() -> None:
    tracking_mode = setup_mlflow_tracking()
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    print(f"Data latih : {X_train.shape} | Data uji: {X_test.shape}")

    # ---- Hyperparameter tuning (kriteria Skilled) ----
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = round(time.time() - start, 2)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "train_accuracy": accuracy_score(y_train, best_model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_cv_f1": search.best_score_,
        "tuning_time_seconds": elapsed,
    }

    # ---- Siapkan artefak tambahan (di luar cakupan autolog) ----
    os.makedirs(TMP_DIR, exist_ok=True)
    save_confusion_matrix(y_test, y_pred, os.path.join(TMP_DIR, "confusion_matrix.png"))
    save_feature_importance(best_model, X_train.columns, os.path.join(TMP_DIR, "feature_importance.png"))
    save_classification_report(y_test, y_pred, os.path.join(TMP_DIR, "classification_report.txt"))

    # ---- MANUAL logging ke MLflow (setara autolog + tambahan) ----
    with mlflow.start_run(run_name="random-forest-tuned-manual") as run:
        # 1) Parameter
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "tuning_method": "RandomizedSearchCV",
                "n_iter": N_ITER,
                "cv_folds": CV_FOLDS,
                "scoring": SCORING,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_train_rows": X_train.shape[0],
                "n_features": X_train.shape[1],
                **search.best_params_,
            }
        )

        # 2) Metrik (accuracy, precision, recall, f1, roc_auc, ...)
        mlflow.log_metrics(metrics)

        # 3) Model dengan signature & input example
        signature = infer_signature(X_train, best_model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # 4) Artefak tambahan (confusion matrix, feature importance, classification report)
        for fname in ("confusion_matrix.png", "feature_importance.png", "classification_report.txt"):
            mlflow.log_artifact(os.path.join(TMP_DIR, fname), artifact_path="artifacts")

        print("\n=== Hasil tuning ===")
        print("Best params      :", search.best_params_)
        for key, value in metrics.items():
            print(f"{key:<22}: {round(value, 4)}")
        print("\nRun ID     :", run.info.run_id)
        print("Model URI  :", model_info.model_uri)
        print("Tracking   :", "DagsHub (online)" if tracking_mode == "dagshub" else "lokal ./mlruns")
        print("Cek UI     : mlflow ui --port 5000  ->  http://localhost:5000")
        print("\nContoh serving model ini:")
        print(f"  mlflow models serve -m '{model_info.model_uri}' -p 5001 --env-manager local")

    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
