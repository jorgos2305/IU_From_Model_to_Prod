import pandas as pd
import mysql.connector
import os
from pathlib import Path
from dotenv import load_dotenv

BASEPATH = Path(__file__).resolve().parents[2]
config_loaded = load_dotenv(BASEPATH / ".env")
if not config_loaded:
    raise ValueError(f"No environment variables found under {BASEPATH}. Check file location")

mysql_config = {
    "user"      : os.getenv("DB_USER"),
    "password"  : os.getenv("DB_PASSWORD"),
    "database"  : "turbine",
    "host"      : os.getenv("DB_SERVER"),
    "port"      : int(os.getenv("DB_PORT", "3306")),
}

path_to_training_data = BASEPATH /"app" / "models" / "training_data" / "turbine_anomalies.csv"

df = pd.read_csv(path_to_training_data)
records = df.to_records(index=False).tolist()

sql = """
INSERT INTO prediction (station_id, ts, temperature, humidity, noise, is_anomaly, y_true, created_at, mlflow_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

with mysql.connector.connect(**mysql_config) as conn:
    with conn.cursor() as cursor:
        try:
            cursor.executemany(sql, records)
            conn.commit()
        except mysql.connector.Error as error:
            conn.rollback()
            print(f"Unexpected error: {error}")
        else:
            print("Records inserted into database")