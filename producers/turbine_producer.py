import time
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Producer, Message, KafkaError
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
            print(f"[INFO] Producer stopped by user - {datetime.now().replace(microsecond=0).isoformat()}")
        finally:
            self.producer.flush()

    def delivery_callback(self, error:KafkaError | None, msg:Message):
        timestamp_type, timestamp = msg.timestamp()
        timestamp = datetime.fromtimestamp(timestamp / 1000).isoformat()

        if error:
            print(f"[ERROR] Message delivery failed: {error} - {timestamp}")
        else:
            print(f"[INFO] Message with key: {msg.key()} delivered to Topic: {msg.topic()} Partition:{msg.partition()} Offset:{msg.offset()} - Timestamp[{timestamp_type}]:{timestamp}")


if __name__ == "__main__":

    BASEPATH = Path(__file__).resolve().parents[1]
    
    config_loaded = load_dotenv(BASEPATH / ".env")
    if not config_loaded:
        raise ValueError(f"No environment variables found under {BASEPATH}. Check file location")

    # Config of kafka producer
    config = {
        "bootstrap.servers" : os.getenv("BOOTSTRAP_SERVERS"),
        "client.id" : "turbine_producer_pi1",
        "acks":-1
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