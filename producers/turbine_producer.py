import numpy as np
import json
from confluent_kafka import Producer

from sensors.production_line import MeasurementStation

class TurbineProducer:
    
    def __init__(self, station:MeasurementStation) -> None:
        self.station = station

    def get_measurement(self):
        return self.station.get_measurement()

if __name__ == "__main__":
    
    station = MeasurementStation("Line-001")
    producer = TurbineProducer(station)

    print(producer.get_measurement())