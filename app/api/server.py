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

class PredictionReponse(BaseModel):
    station_id : str
    timestamp : datetime
    is_anomaly : bool
    y_true : bool

# TODO: Setup resources on start up: Database, Model

# database

def insert_prediction(measurement, is_anomaly):
    print({"measurement" : measurement, "is_anomaly" : is_anomaly})

# Load anomaly detector
model = TurbineAnomalyDetector()

app = FastAPI()

# endpoints

@app.post("/predict/")
def predict(measurement:MeasurementData):
    X = [[measurement.temperature, measurement.humidity, measurement.noise]]
    anomaly_score = model.predict(X)[0]
    # if is anomaly the value returned by the Isolation forest is -1 otherwise 1
    is_anomaly = anomaly_score == -1

    # record prediction in DB
    insert_prediction(measurement, is_anomaly)

    return PredictionReponse(
        station_id=measurement.station_id,
        timestamp=measurement.timestamp,
        is_anomaly=is_anomaly,
        y_true=measurement.is_anomaly
    )

@app.post("/model/reload/")
def reload_model():
    model.reload()
    return {"timestamp":_now(), "status": "reloaded"}

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

# helpers

def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()