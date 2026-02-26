import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List

class TurbineAnomalyDetector:

    def __init__(self) -> None:
        pass

    def predict(self, X:np.ndarray | pd.DataFrame | List) -> np.ndarray:
        raise NotImplementedError
    
    def load(self, model_path:Path) -> None:
        raise NotImplementedError
    
    def get_params(self) -> Dict:
        raise NotImplementedError