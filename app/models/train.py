import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import mysql.connector
from typing import Dict, Tuple, List

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 1. set up training

# make sure env is found
BASEPATH = Path(__file__).resolve().parents[2]
config_loaded = load_dotenv(BASEPATH / ".env")
if not config_loaded:
    raise ValueError(f"No environment variables found under {BASEPATH}. Check file location")

# mlflow config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "TurbineAnomalyDetector_training"
if not MLFLOW_TRACKING_URI:
    raise ValueError(f"MLFLOW_TRACIKING_URI does not exist. Check correct server: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. set up database

mysql_config = {
    "user"      : os.getenv("DB_USER"),
    "password"  : os.getenv("DB_PASSWORD"),
    "database"  : "turbine",
    "host"      : os.getenv("DB_SERVER"),
    "port"      : int(os.getenv("DB_PORT", "3306")),
}

# 3. train

def get_training_data(n_samples:int=15_000, config=mysql_config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    sql = """
    SELECT temperature, humidity, noise, y_true
    FROM turbine.prediction
    ORDER BY ts DESC
    LIMIT %s
    """
    try:
        connection = mysql.connector.connect(**config)
        with connection as conn:
            df = pd.read_sql(sql, con=connection, params=(n_samples,))     # type: ignore
    except mysql.connector.errors.Error as mysql_error:
        raise ValueError(f"[ERROR] Database error: {mysql_error}")
    except Exception as exc:
        raise ValueError(f"[ERROR] Unexpected error: {exc}")
    if df.empty:
        raise ValueError("[ERROR] No training data found in the database")
    
    db_samples, _ = df.shape
    if db_samples < n_samples:
        print(f"[WARNING] Only {db_samples} available in database, requested {n_samples}")
    
    X = df.drop(columns="y_true")
    y = df["y_true"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    return X_train, X_test, y_train, y_test

def evaluate(y_true:pd.Series|np.ndarray, y_pred:pd.Series|np.ndarray) -> Dict:
    return {
        "precision" : precision_score(y_true, y_pred),
        "recall" : recall_score(y_true, y_pred),
        "f1_score" : f1_score(y_true, y_pred)
    }
    
def train() -> str:

    # Get the data
    print("[INFO] retrieving training data from database")
    X_train, X_test, y_train, y_test = get_training_data()

    # register training run
    with mlflow.start_run(run_name="Train_new_model") as run:

        print(f"[INFO] Setting up MLflow run: {run.info.run_id}")   
        # params definition
        anomaly_detector_params = {
            "n_estimators" : 700,
            "contamination" : 0.06,
            "bootstrap":True,
            "contamination_train" : y_train.mean(),
            "contamination_test" : y_test.mean()
        }
        mlflow.log_params(anomaly_detector_params)

        # train new model
        print(f"[INFO] Training TurbineAnomalyDetector")
        anomaly_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("iForest", IsolationForest(n_estimators=anomaly_detector_params["n_estimators"],
                                            contamination=anomaly_detector_params["contamination"],
                                            bootstrap=anomaly_detector_params["bootstrap"]))
                ]
            )
        anomaly_pipeline.fit(X_train)

        # add model to the registry
        signature = infer_signature(X_train.iloc[:5,:], anomaly_pipeline.score_samples(X_train.iloc[:5,:]))
        model_info = mlflow.sklearn.log_model( # type: ignore
            sk_model=anomaly_pipeline,
            name="TurbineAnomalyDetector",
            signature=signature
            )

        # Use custom evaluation
        y_train_pred = np.where(
            anomaly_pipeline.predict(X_train) == -1, 1, 0 # convention used y = 1 anomaly; y = 0 no anomaly
        )
        # avoid overwriting the metrics of the test and train sets
        train_metrics = {
            f"train_{key}" : value
            for key, value
            in evaluate(y_true=y_train, y_pred=y_train_pred).items()
            }
        mlflow.log_metrics(train_metrics)

        y_test_pred = np.where(
           anomaly_pipeline.predict(X_test) == -1, 1, 0 # convention used y = 1 anomaly; y = 0 no anomaly
        )
        test_metrics = {
            f"test_{key}" : value
            for key, value
            in evaluate(y_true=y_test, y_pred=y_test_pred).items()
        }
        mlflow.log_metrics(test_metrics)

    return model_info.model_uri
    
if __name__ == "__main__":
    # quick tests
    print(train())
