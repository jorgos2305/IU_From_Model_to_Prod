from datetime import datetime
import pprint
import mlflow
from mlflow.client import MlflowClient
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Any

from sklearn.pipeline import Pipeline

BASEPATH = Path(__file__).resolve().parents[2]
config_loaded = load_dotenv(BASEPATH / ".env")
if not config_loaded:
    raise ValueError(f"No environment variables found under {BASEPATH}. Check file location")

# mlflow config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise ValueError(f"MLFLOW_TRACKING_URI does not exist. Check correct server: {MLFLOW_TRACKING_URI}")
EXPERIMENT_NAME = "TurbineAnomalyDetector_training"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

class TurbineAnomalyDetector:

    def __init__(self) -> None:
        self._name = "TurbineAnomalyDetector"
        self._alias = "champion"
        self._client = MlflowClient()
        self._model_uri = f"models:/{self._name}@{self._alias}"
        self._pipeline = self._load_model()
        
    def predict(self, X:List) -> int:
        return self._pipeline.predict(X)[0] # type: ignore
    
    def _load_model(self) -> Pipeline:
        model = mlflow.sklearn.load_model(self._model_uri)  # type: ignore
        self._model_info = self._client.get_model_version_by_alias(self._name, self._alias)
        return model # type: ignore
    
    def reload(self) -> None:
        print("[INFO] Loading new model")
        self._pipeline = self._load_model()
    
    def get_params(self) -> Dict[str, Any]:
        params = self._pipeline.named_steps["iForest"].get_params()
        aliases = [str(a) for a in self._model_info.aliases]
        return {
            "model_name" : self._model_info.name,
            "model_aliases" : aliases,
            "model_version" : self._model_info.version,
            "n_estimators" : params["n_estimators"],
            "contamination" : params["contamination"],
            "bootstrap" : params["bootstrap"]
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        aliases = [str(a) for a in self._model_info.aliases]
        return {
            "model_name" : self._model_info.name,
            "model_aliases" : aliases,
            "model_version" : self._model_info.version
        }

if __name__ == "__main__":

    try:
        print("Loading model")
        model = TurbineAnomalyDetector()
        print("Model loaded")
        pprint.pp(model.get_params(), indent=4)
        model.reload()
    except Exception as exc:
        print(f"Unable to load model: {exc}")
        print(BASEPATH)
