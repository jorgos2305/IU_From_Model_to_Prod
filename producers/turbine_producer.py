import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Producer
from confluent_kafka import Message, KafkaError
from typing import Dict

from sensors.production_line import MeasurementStation

class TurbineProducer:
    
    def __init__(self, station:MeasurementStation, config:Dict, topic:str) -> None:
        self.station = station
        self.producer = Producer(config)
        self._topic = topic

    def get_measurement(self) -> str:
        measurement = self.station.get_measurement()
        return json.dumps(measurement)
    
    def produce(self) -> None:
        try:
            while True:
                self.producer.produce(
                    topic=self._topic,
                    key=b"pi_1",
                    value=self.get_measurement().encode("utf-8"),
                    callback=self.delivery_callback
                )
                self.producer.poll(0)
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"[INFO] Producer stopped by user - {self._now()}")
        finally:
            self.producer.flush()

    def delivery_callback(self, error:KafkaError | None, msg:Message):
        if error:
            print(f"[ERROR] Message delivery failed: {error} - {msg.timestamp()}")
        else:
            print(f"[INFO] Message {msg} delivered to Topic: {msg.topic()} Partition:{msg.partition()} Offset:{msg.offset()} - {msg.timestamp()}")
    
    def _now(self) -> str:
        return datetime.now().replace(microsecond=0).isoformat()


if __name__ == "__main__":

    load_dotenv(r"IU_From_Model_to_Prod/.env")

    # Config of kafka producer
    config = {
        "bootstrap.servers" : os.getenv("BOOTSTRAP_SERVERS"),
        "client.id" : "turbine_producer_pi1"
    }
    topic = "turbine_pi1"
    
    # get measurement data
    station = MeasurementStation("pi1")
    # set up producer and produces messages
    prod = TurbineProducer(
        station=station,
        config=config,
        topic=topic
    )
    prod.produce()