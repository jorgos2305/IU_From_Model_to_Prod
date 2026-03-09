# IU International University of Applied Sciences

## Course: From Model To Production (DLBDSMTP01)
Welcome to the repository of the __From Model to Production__ course.  

The task for this project consists in building a simple machine learning model that uses temperature, humidty and noise sensor data to detect anomalies during the production of wind turbine components. The Focus of the system is to design a model that is easily implemented in the productive system.

The system must be able to process data streams, since the sensors measure data continuously. The model should take these measurements over a standarized API and respond with a prediction score for an anomaly.
The system should be monitorable, maintainable, scalable and adaptable to new data.

### Objectives

## Dataset information

## Folder structure

```bash
.
├── README.md                           # Main project documentation
└── app                                 # Main application package
    ├── __init__.py                     # Makes app a Python module
    ├── api                             # Conatains modules used for the API
    │   ├── __init__.py
    │   └── server.py                   # REST API implementation
    ├── consumers                       # Contains modulels used for the Kafka consumer
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
        ├── production_line.py          # Handle the genration of simulated sensor data
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

### Apache Kafka

### MLflow

