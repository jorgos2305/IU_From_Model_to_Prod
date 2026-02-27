import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Any

from sklearn.pipeline import Pipeline

BASEPATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASEPATH / "models" / "trained_models" / "anomaly_detector.joblib"


class TurbineAnomalyDetector:

    def __init__(self, model_path:Path=MODEL_PATH) -> None:
        self._model_path = model_path
        self._pipeline : Pipeline = self._load_model() # type avoid the pylance warning

    def predict(self, X:List) -> np.ndarray:
        return self._pipeline.predict(X) # type: ignore
    
    def _load_model(self) -> Pipeline:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train a new model with train.py to generate a new model.")
        
        try:
            pipeline = joblib.load(self._model_path)
            print(f"[INFO] Model loaded sucessfully: {self._model_path}")
        except Exception as exc:
            raise RuntimeError(f"Unable to load model {self._model_path} - {exc}")
        else:
            return pipeline
    
    def reload(self) -> None:
        print("Loading new model")
        self._pipeline = self._load_model()
    
    def get_params(self) -> Dict[str, Any]:
        return self._pipeline.get_params()


if __name__ == "__main__":

    try:
        model = TurbineAnomalyDetector()
    except Exception as exc:
        print(f"Unable to load model: {exc}")
        print(BASEPATH)
    else:
        print("model loaded")
