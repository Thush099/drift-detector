from typing import List
import logging
from src.models import MachineLearningModel
from src.utils import calculate_drift

class DriftDetectorPipeline:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.model = MachineLearningModel()

    def fit(self, data: List[float]):
        self.model.train(data[:self.window_size])

    def predict(self, data: List[float]):
        predictions = []
        for i in range(self.window_size, len(data)):
            prediction = self.model.predict(data[i])
            predictions.append(prediction)
            drift = calculate_drift(data[i-self.window_size:i], prediction)
            if drift:
                logging.info('Drift detected at time step %d', i)
                self.model.retrain(data[i-self.window_size:i])
        return predictions