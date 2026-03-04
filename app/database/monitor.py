import mlflow
from mlflow.client import MlflowClient
import time
import os
import mysql.connector
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, Tuple
from scipy.stats import ttest_ind

from sklearn.pipeline import Pipeline

from app.models.train import train

class MonitorService:

    def __init__(self, db_config:Dict, mlflow_tracking_uri:str, api_address:str, check_interval:int, n_samples:int, alpha:float) -> None:
        self._db_config = db_config
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._api_address = api_address
        self._check_interval = check_interval
        self._n_samples = n_samples
        self._alpha = alpha
        self._model_name = "TurbineAnomalyDetector"
        self._model_alias = "champion"
    
    def run_check(self) -> None:
        # when staring get the data the model was trained on
        min_id, max_id = self._get_champion_ids()
        from_id = max_id
        try:
            while True:
                most_recent_data = self._get_most_recent_data(from_id)
                n_rows, _ = most_recent_data.shape
                if n_rows < self._n_samples: # not enough data for retraining
                    print(f"[INFO] Not enough data for training. Found: {n_rows} Requested:{self._n_samples}")
                    print(f"[INFO] Waiting for {self._check_interval} seconds")
                    time.sleep(self._check_interval)
                    continue
                print(f"[INFO] Enough data for training deteted: {n_rows}")
                print(f"[INFO] Start drift check")
                training_data = self._get_reference_data(min_id, max_id)
                has_drifted = self._check_drift(training_data, most_recent_data.drop(columns="id"))
                if has_drifted:
                    self._trigger_retrain()
                    min_id, max_id = self._get_champion_ids()
                    from_id = max_id
                    print(f"[INFO] Retrain triggered. New champion: min_id={min_id}, max_id={max_id}")
                else:
                    from_id = int(most_recent_data["id"].iloc[-1])
                    print(f"[INFO] No drift detected. New id for most recent data: {from_id}")
                time.sleep(self._check_interval)
        except KeyboardInterrupt:
            print(f"[INFO] Monitor service stopped by user")

    def _get_champion_ids(self) -> Tuple[int, int]:
        client = MlflowClient()
        model_info = client.get_model_version_by_alias(self._model_name, self._model_alias)
        min_id = int(model_info.tags["min_id"])
        max_id = int(model_info.tags["max_id"])
        return min_id, max_id
    
    def _get_reference_data(self, min_id:int, max_id:int) -> pd.DataFrame:
        sql = """
        SELECT temperature, humidity, noise
        FROM turbine.prediction
        WHERE id BETWEEN %s AND %s
        ORDER BY id ASC
        LIMIT %s
        """
        try:
            with mysql.connector.connect(**self._db_config) as conn:
                df = pd.read_sql(sql, con=conn, params=(min_id, max_id, self._n_samples)) # type: ignore
        except mysql.connector.Error as mysql_error:
            raise ValueError(f"[ERROR] Database error: {mysql_error}")
        except Exception as exc:
            raise ValueError(f"[ERROR] Unexpected error: {exc}")
        return df
    
    def _get_most_recent_data(self, from_id:int) -> pd.DataFrame:
        sql = """
        SELECT id, temperature, humidity, noise
        FROM turbine.prediction
        WHERE id > %s
        ORDER BY id ASC
        LIMIT %s
        """
        try:
            with mysql.connector.connect(**self._db_config) as conn:
                df = pd.read_sql(sql, con=conn, params=(from_id, self._n_samples)) # type: ignore
        except mysql.connector.Error as mysql_error:
            raise ValueError(f"[ERROR] Database error: {mysql_error}")
        except Exception as exc:
            raise ValueError(f"[ERROR] Unexpected error: {exc}")
        return df
    
    def _check_drift(self, df1:pd.DataFrame, df2:pd.DataFrame) -> bool:
        drifted = False
        for col in df1.columns:
            result = ttest_ind(df1[col], df2[col], equal_var=False, alternative="two-sided")
            drifted = result.pvalue <= self._alpha # type: ignore
            if drifted:
                print(f"[INFO] Drift detected in '{col}' (p={result.pvalue:.6f}) alpha={self._alpha})") # type: ignore
                return drifted
        return drifted
    
    def _trigger_retrain(self) -> None:
        try:
            train()
        except Exception as exc:
            raise RuntimeError(f"[ERROR] New model cannot be trained: {exc}")

if __name__ == "__main__":
    
    # make sure env is found
    BASEPATH = Path(__file__).resolve().parents[2]
    config_loaded = load_dotenv(BASEPATH / ".env")
    if not config_loaded:
        raise ValueError(f"No environment variables found under {BASEPATH}. Check file location")
    
    # mlflow config
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        raise ValueError(f"MLFLOW_TRACIKING_URI does not exist. Check correct server: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 2. set up database
    mysql_config = {
        "user"      : os.getenv("DB_USER"),
        "password"  : os.getenv("DB_PASSWORD"),
        "database"  : "turbine",
        "host"      : os.getenv("DB_SERVER"),
        "port"      : int(os.getenv("DB_PORT", "3306")),
    }
    # 3. API info
    API_ADDRESS = os.getenv("API_ADDRESS")
    if not API_ADDRESS:
        raise ValueError("No API Information available")
    
    monitor = MonitorService(
        mysql_config,
        MLFLOW_TRACKING_URI,
        API_ADDRESS,
        300, # change this for testing
        5000, # change for testing
        0.05
    )
    monitor.run_check()