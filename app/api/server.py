from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd

# load model
model = joblib.load(r"app/models/trained_models/anomaly_detector.joblib")

class MeasurementData(BaseModel):
    station_id : str
    timestamp : str
    temperature : float
    humidity : float
    noise : float
    is_anomaly : bool

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello" : "World"}

@app.post("/predict/")
def predict(measurement:MeasurementData):
    X = pd.DataFrame(
        {
            "temperature" : [measurement.temperature],
            "humidity" : [measurement.humidity],
            "noise" : [measurement.noise]
            }
        )
    anomaly_score = model.predict(X)[0]
    if anomaly_score == -1:
        is_anomaly = True
    else:
        is_anomaly = False
    insert_to_database(measurement, is_anomaly)
    return {"is_anomaly" : is_anomaly, "y_true" : measurement.is_anomaly}

def insert_to_database(measurement, is_anomaly):
    print({"measurement" : measurement, "is_anomaly" : is_anomaly})

@app.get("/health/")
def health_check():
    return {"Service is healthy" : True}