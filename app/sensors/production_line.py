import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any

from app.sensors.utils import SensorType

class SensorIoT:

    def __init__(self, mean:int|float, std:int|float, kind:SensorType, seed:int|None=None) -> None:
        self.random_generator = np.random.default_rng(seed)
        self.mean = mean
        self.std  = std
        self._kind = kind

    def generate_reading(self, anomaly_rate=0.02) -> Tuple[float, bool]:
        reading = self.random_generator.normal(self.mean, self.std)
        # use for evaluating model performance
        is_anomaly = False
        if self.random_generator.random() < anomaly_rate:
            match self._kind:
                case SensorType.TEMPERATURE:
                    mean_anomaly = self.random_generator.choice([-10, 10])
                    anomaly = self.random_generator.normal(mean_anomaly, 2)
                    a_min, a_max = -10, 50
                case SensorType.HUMIDITY:
                    mean_anomaly = self.random_generator.choice([-35, 35])
                    anomaly = self.random_generator.normal(mean_anomaly, 5)
                    a_min, a_max = -10, 100
                case SensorType.NOISE:
                    mean_anomaly = self.random_generator.choice([-25, 35])
                    anomaly = self.random_generator.normal(mean_anomaly, 5)
                    a_min, a_max = 0, 120
                case _:
                    anomaly, a_min, a_max = 0, 0 ,0
            is_anomaly = True
            return np.clip(reading + anomaly, a_min=a_min, a_max=a_max), is_anomaly
        return reading, is_anomaly
    
    @property
    def kind(self) -> SensorType:
        return self._kind

class MeasurementStation:

    def __init__(self, station_id:str, seed:int|None=None) -> None:
        self.station_id = station_id
        self.sensors = [
            SensorIoT(24, 2, SensorType.TEMPERATURE, seed),
            SensorIoT(45, 7, SensorType.HUMIDITY, seed),
            SensorIoT(50, 6, SensorType.NOISE, seed)
        ]

    def get_measurement(self) -> Dict[str, Any]:
        measurement : Dict[str, Any]= { # this removes the pylance warning
            "station_id" : self.station_id,
            "timestamp" : datetime.now().replace(microsecond=0).isoformat()
        }
        
        is_any_anomaly = False
        
        for sensor in self.sensors:
            reading, is_anomaly = sensor.generate_reading()
            # flag if there is any anomaly in a sensor
            is_any_anomaly = is_any_anomaly or is_anomaly # avoid overwritting in case the first sensor is anomaly and the last one is not an anomaly
            measurement[sensor.kind.value] = reading
        measurement["is_anomaly"] = is_any_anomaly
        
        return measurement

if __name__ == "__main__":
    # quick functional tests

    from time import sleep

    station = MeasurementStation("test_station")
    while True:
        try:
            print(station.get_measurement())
            sleep(1)
        except KeyboardInterrupt:
            print("test finished")
            break

    