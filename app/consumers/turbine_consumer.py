import json
import os
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from confluent_kafka import Consumer, Message
from typing import Dict
import logging

from app.logging_config import setup_logging

class TurbineConsumer:

    def __init__(self, config:Dict, topic:str, api:str) -> None:
        self._config = config
        self.consumer = Consumer(config)
        self.consumer.subscribe([topic])
        self._api = api
        self._logger = logging.getLogger(__name__)

    def consume(self) -> None:
        self._logger.info("Start to consume data from Kafka")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    # No message has been delivered yet, try again
                    continue
                err = msg.error()
                if err:
                    # In case of Kafka error, add to log
                    self._logger.error(f"Consumer error: {err.code()} - {err.str()}")
                else:
                    self.process(msg) # Send the message to the FastAPI application
        except KeyboardInterrupt:
            self._logger.info(f"Consumer stopped by user - {datetime.now().replace(microsecond=0).isoformat()}")
        finally:
            self.consumer.close()
    
    def process(self, msg:Message) -> None:
        value_bytes = msg.value()
        if not value_bytes:
            self._logger.error(f"Message with {msg.key()} contains no data.")
            return
        try:
            data = json.loads(value_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            self._logger.error(f"Message does not contain a valid JSON document.")
            return
        except UnicodeDecodeError:
            self._logger.error(f"Message does not contain valid UTF-8 encoded data.")
            return
        try:
            response = requests.post(f"{self._api}/predict/", json=data)
            response.raise_for_status()
            self._logger.error(f"Received response status code: {response.status_code}. {response.text}")
        except requests.HTTPError:
            self._logger.error(f"Unable to place request for prediction.")
            return

if __name__ == "__main__":

    # setup logging
    setup_logging(log_file="consumer.log")

    # locate necessary resources
    BASEPATH = Path(__file__).resolve().parents[2]
    
    # Load environment variables
    config_loaded = load_dotenv(BASEPATH / ".env")
    if not config_loaded:
        raise ValueError(f"No environment variable found under {str(BASEPATH)}. Check file location.")
    
    bootstrap_servers = os.getenv("BOOTSTRAP_SERVERS")
    api_address = os.getenv("API_ADDRESS")
    if bootstrap_servers is None or api_address is None:
        raise ValueError("Unable to load environment variables.")

    # kafka consumer config
    config = {
        "bootstrap.servers": bootstrap_servers,
        "client.id" : "turbine_consumer_pi1",
        "group.id" : "turbine_consumers",
        "auto.offset.reset" :"earliest",
        "enable.auto.commit" : True
        }
    
    # Ensure the topic is the same as the one the producer writes to
    topic = "turbine_pi1"

    # set up the Turbine Consumer start fetching messages from kafka
    consumer = TurbineConsumer(
        config=config,
        topic=topic,
        api=api_address
    )
    consumer.consume()