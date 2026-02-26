from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

from app.models.anomaly_detector import TurbineAnomalyDetector

class MeasurementData(BaseModel):
    """
    Defines the structure of the data used to make a prediction.
    """
    station_id : str
    timestamp : datetime
    temperature : float
    humidity : float
    noise : float
    is_anomaly : bool

model = TurbineAnomalyDetector()

app = FastAPI()

@app.post("/predict/")
def predict(measurement:MeasurementData):
    X = [[measurement.temperature, measurement.humidity, measurement.noise]]
    anomaly_score = model.predict(X)[0]
    if anomaly_score == -1:
        is_anomaly = True
    else:
        is_anomaly = False
    insert_to_database(measurement, is_anomaly)
    return {"is_anomaly" : is_anomaly, "y_true" : measurement.is_anomaly}

def insert_to_database(measurement, is_anomaly):
    print({"measurement" : measurement, "is_anomaly" : is_anomaly})

@app.get("/model/info/")
def model_info():
    return {
        "timestamp" : _now(),
        "model_params" : model.get_params()
    }

@app.get("/health/")
def health_check():
    return {
        "timestame" : _now(),
        "is_healthy" : True
        }

# Helper functions
def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()