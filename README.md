# IU International University of Applied Sciences

## Course: From Model To Production (DLBDSMTP01)
Welcome to the repository of the __From Model to Production__ course.  

The task for this project consists in building a simple machine learning model that uses temperature, humidity and noise sensor data to detect anomalies during the production of wind turbine components. The focus of the project is to design a model that is easily implemented in the productive system.

The system must be able to process data streams, since the sensors measure data continuously. The model should take these measurements over a standardized API and respond with a prediction score for an anomaly.
The system should be monitorable, maintainable, scalable and adaptable to new data.

### Objective

To implement a system that:

1. Processes continuous IoT sensor streams
2. Detects anomalies in production
3. Integrates a machine learning model into a production pipeline
4. Serves predictions over a REST API
5. Support monitoring, scalability, maintainability and traceability

## Dataset information

- The project uses simulated sample data
- The features include Temperature, Humidity and Noise
- An example of the dataset is found under ```app/models/training_data/turbine_anomalies.csv```

## Folder structure

```bash
.
├── README.md                           # Main project documentation
└── app                                 # Main application package
    ├── __init__.py                     # Makes app a Python module
    ├── api                             # Contains modules used for the API
    │   ├── __init__.py
    │   └── server.py                   # REST API implementation
    ├── consumers                       # Contains modules used for the Kafka consumer
    │   ├── __init__.py
    │   └── turbine_consumer.py         # Implementation of the Kafka consumer. Stream processing
    ├── database                        # Contains components needed to setup the database
    │   ├── __init__.py
    │   ├── monitor.py                  # Monitoring service, handles drift detection
    │   └── turbine.sql                 # SQL script for creating the database
    ├── models                          # Contains modules needed for the ML model
    │   ├── __init__.py
    │   ├── anomaly_detector.py         # Implements TurbineAnomalyDetector
    │   ├── first_model.py              # Populates database. Needed once for setting up the database
    │   ├── train.py                    # Training service
    │   └── training_data
    │       └── turbine_anomalies.csv   # Training data needed to build the first TurbineAnomalyDetector
    ├── producers                       # Contains modules needed for the Kafka producers
    │   ├── __init__.py
    │   └── turbine_producer.py         # Implementation of the Kafka producer. Stream generation
    └── sensors                         # Data simulation components
        ├── __init__.py
        ├── production_line.py          # Handle the generation of simulated sensor data
        └── utils.py                    # Utilities used by the SensorIoT class
```

## How to run the system locally

### Conda environment
This project was implemented using a conda environment. To replicate it, run the following commands:

```bash
git clone https://github.com/jorgos2305/IU_From_Model_to_Prod.git
cd IU_From_Model_to_Prod
conda env create -f environment.yml
conda activate mlops
```

### Important

The system uses:
- __MySQL__ for data persistence
- __Apache Kafka__ to handle data streaming
- __MLflow__ for the model lifecycle management.  

These instructions assume you have already installed all necessary dependencies.

Assuming all components run locally on your machine, the default settings of each component are sufficient.

### MySQL

In your terminal, run the following command and enter your password when prompted:

```bash
mysql -u root -p -h 127.0.0.1 < app/database/turbine.sql
```

### Apache Kafka

The project uses a recent Kafka version and does not utilize Zookeeper, it runs in __KRaft mode__.

For more details see:  
[KRaft vs. Zookeeper](https://kafka.apache.org/42/getting-started/zk2kraft/).

To setup the Kafka topic follow the steps below:

1. In your terminal go to the folder where your Kafka installation is located
2. Enter the following command to setup the topic:

```bash
bin/kafka-topics.sh --create \
-- bootstrap-server localhost:9092 \
-- topic turbine_p1 \
-- partitions 1 \
-- replication-factor 1 \
-- config min.insync.replicas=1 \
-- config retention.ms=86400000
```
3. Once the Topic has been set up, run the following to start Kafka:

```bash
bin/kafka-server-start.sh config/kraft/server.properties 
```

### MLflow

Open a new terminal and make sure you are in the correct conda environment.  
Once the environment is activated, run:

```bash
mlflow server --port 5000
```

### Anomaly Detection System

The next step after setting up MySQL, Kafka and MLflow is to ensure that the training data is correctly stored in the database.

The simulated data can be found under ```app/models/training_data/turbine_anomalies.csv```.

To insert the data from the ```.csv```file into the database run:

```bash
python -m app.models.first_model
```

Once the records have been inserted into the database, run the following commands to start the system

```bash
python -m app.models.train
fastapi dev app/api/server.py
python -m app.database.monitor              # Run in new terminal window
python -m app.consumers.turbine_consumer    # Run in new terminal window
python -m app.producers.turbine_producer    # Run in new terminal window
```

