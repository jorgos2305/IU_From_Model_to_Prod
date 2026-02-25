from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

class MeasurementData(BaseModel):
    station_id : str
    timestamp : str
    temperature : float
    humidity : float
    is_anomaly : bool

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello" : "World"}

@app.post("/predict/")
def create_prediction(measurement:MeasurementData) -> Dict:
    """
    Runs anomaly detector on the measurement and storos both the measurement and the prediction of the result
    in the database.

    Args:
        measurement (MeasurementData): Measurement data from the sensor containing:\n
        1. station_id
        2. timestamp
        3. temperature
        4. humidity
        5. is_anomaly -> Whether the simulated data is an anomaly

    Returns:
        Dict: Whether data was correctly received.
    """
    print("Measurement received")
    return {"status":"success", "received_data":measurement}