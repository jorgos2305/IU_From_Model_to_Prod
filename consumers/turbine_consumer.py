import time
import json
import os
from datetime import datetime
from confluent_kafka import Consumer, Message, KafkaError
from typing import Dict, List

class TurbineConsumer:

    def __init__(self, config:Dict, topic:str) -> None:
        self._config = config
        self.consumer = Consumer(config)
        self._topic = topic
        self.consumer.subscribe([topic])

    def consume(self) -> None:
        try:
            while True:
                msg = self.consumer.poll(1)
                if msg is not None and msg.error() is None:
                    print(msg.value())
                    
        except:
            print("error")
    
    def process(self, msg:Message) -> None:
        value_bytes = msg.value()
        if not value_bytes:
            print(value_bytes)
        else:
            print(value_bytes.decode("utf-8"))

if __name__ == "__main__":

    config = {
        "bootstrap.servers":"192.168.0.157:9092",
        "client.id" : "turbine_consumer_pi1",
        "group.id" : "my_group"
        }
    
    topic = "turbine_pi1"

    consumer = TurbineConsumer(
        config=config,
        topic=topic
    )
    consumer.consume()