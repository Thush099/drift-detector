from typing import List
import logging
from sklearn.linear_model import LinearRegression

class MachineLearningModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, data: List[float]):
        self.model.fit([[i] for i in range(len(data))], data)

    def predict(self, data: float):
        return self.model.predict([[data]])[0]

    def retrain(self, data: List[float]):
        self.model.fit([[i] for i in range(len(data))], data)