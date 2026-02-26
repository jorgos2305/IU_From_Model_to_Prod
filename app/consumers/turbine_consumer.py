import json
import os
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Consumer, Message
from typing import Dict

class TurbineConsumer:

    def __init__(self, config:Dict, topic:str) -> None:
        self._config = config
        self.consumer = Consumer(config)
        self.consumer.subscribe([topic])

    def consume(self) -> None:
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    # No message has been delivered yet, try again
                    continue
                err = msg.error()
                if err:
                    print(f"[KafkaError] Consumer error: {err.code()} - {err.str()}")
                else:
                    self.process(msg) # Send the message to the FastAPI application
        except KeyboardInterrupt:
            print(f"[INFO] Consumer stopped by user - {datetime.now().replace(microsecond=0).isoformat()}")
        finally:
            self.consumer.close()
    
    def process(self, msg:Message) -> None:
        value_bytes = msg.value()
        if not value_bytes:
            print(value_bytes)
        else:
            data = json.loads(value_bytes.decode("utf-8"))
            print(data)
            response = requests.post("http://127.0.0.1:8000/predict/", json=data)
            print(response.text)

if __name__ == "__main__":

    BASEPATH = Path(__file__).resolve().parents[2]
    
    config_loaded = load_dotenv(BASEPATH / ".env")
    if not config_loaded:
        raise ValueError(f"No environment variable found under {BASEPATH}. Check file location.")
    
    # Config of kafka Consumer

    config = {
        "bootstrap.servers": os.getenv("BOOTSTRAP_SERVERS"),
        "client.id" : "turbine_consumer_pi1",
        "group.id" : "turbine_consumers",
        "auto.offset.reset" :"earliest",
        "enable.auto.commit" : True
        }
    
    topic = "turbine_pi1"

    consumer = TurbineConsumer(
        config=config,
        topic=topic
    )
    consumer.consume()