from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
import mysql.connector

from app.models.anomaly_detector import TurbineAnomalyDetector

BASEPATH = Path(__file__).resolve().parents[2]
environment_loaded = load_dotenv(BASEPATH / ".env")

# ----------------- Data schemas -----------------

class MeasurementData(BaseModel):
    """
    Defines the structure of the data used to make a prediction.
    """
    station_id  : str
    timestamp   : datetime
    temperature : float
    humidity    : float
    noise       : float
    is_anomaly  : bool

class PredictionReponse(BaseModel):
    """
    Defines the structure of the data sent in the response
    """
    station_id : str
    timestamp  : datetime
    is_anomaly : bool
    y_true     : bool

# ----------------- database -----------------

# database configuration

mysql_config = {
    "user"      : os.getenv("DB_USER"),
    "password"  : os.getenv("DB_PASSWORD"),
    "database"  : "turbine",
    "host"      : os.getenv("DB_SERVER"),
    "port"      : int(os.getenv("DB_PORT", "3306")),
}
# database connection pool
pool = mysql.connector.pooling.MySQLConnectionPool(pool_name="turbine_pool", pool_size=1, **mysql_config)

def insert_prediction(measurement:MeasurementData, response:PredictionReponse, pool:mysql.connector.pooling.MySQLConnectionPool=pool) -> None:
    """
    Insert prediction data into the MySQL database.
    """
    add_prediction = f"""INSERT INTO prediction (station_id, ts, temperature, humidity, noise, is_anomaly, y_true, created_at)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
    
    values = (measurement.station_id, measurement.timestamp, measurement.temperature, measurement.humidity, measurement.noise, response.is_anomaly, measurement.is_anomaly, _now())

    with pool.get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(add_prediction, values)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                print(f"[ERROR] Unable to insert prediction into the database: {exc}")
    return
    
# ----------------- Load anomaly detector -----------------

model = TurbineAnomalyDetector()

# ----------------- start API-----------------

app = FastAPI()

# ----------------- endpoints -----------------

@app.post("/predict/")
def predict(measurement:MeasurementData):
    X = [[measurement.temperature, measurement.humidity, measurement.noise]]
    anomaly_score = model.predict(X)
    # if is anomaly the value returned by the Isolation forest is -1 otherwise 1
    is_anomaly = anomaly_score == -1
    response = PredictionReponse(
        station_id=measurement.station_id,
        timestamp=measurement.timestamp,
        is_anomaly=is_anomaly,
        y_true=measurement.is_anomaly
    )
    # record prediction in DB
    insert_prediction(measurement, response)
    return response

@app.post("/model/reload/")
def reload_model():
    """
    Reloads TurbineAnomalyDetector

    Returns:
        The timestamo of when reload is triggered and the status of the model
    """
    model.reload()
    return {"request_timestamp":_now(), "model_loaded": model.get_model_info()}

@app.get("/model/info/")
def model_info():
    """
    Provides information about the TurbineAnomalyDetector currently used, including underlying model and hyperparamaters

    Returns:
        _type_: _description_
    """
    return {"request_timestamp" : _now(), "model_params" : model.get_params()}

@app.get("/health/")
def health_check():
    """
    Health check to verify the API is available.

    Returns:
        Time stamp of the health check
    """
    return {"timestame" : _now(), "is_healthy" : True}

# ----------------- helpers -----------------

def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")