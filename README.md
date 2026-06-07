# Drift Detector

Detects real-time concept drift in machine learning predictions using ADWIN and Page-Hinkley algorithms.

## Problem Statement
Concept drift occurs when the underlying distribution of the data changes over time. This can significantly affect the performance of machine learning models.

## Architecture
```
+---------------+
|  Data Stream  |
+---------------+
       |
       |
       v
+---------------+
|  Drift Detector  |
|  (ADWIN & PH)    |
+---------------+
       |
       |
       v
+---------------+
|  Machine Learning|
|  Model            |
+---------------+
```

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage
To run the drift detector with a sample dataset, use:
```bash
python main.py --help
python main.py --data-size 1000 --drift-size 500 --window-size 200
```
Sample output:
```
Drift detected at time step 512
Model performance: 0.85
```

## Design Decisions
The drift detector uses a combination of ADWIN and Page-Hinkley algorithms to detect concept drift in real-time. The machine learning model is retrained whenever a drift is detected.

The `main.py` script provides a command-line interface for running the drift detector with customizable parameters.