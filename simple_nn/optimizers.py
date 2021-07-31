import numpy as np


class SGD:
    def __init__(self, learning_rate = 1e-4):
        self.learning_rate = learning_rate

    def optimize(self, values, values_grad):
        for value, value_grad in zip(values, values_grad):
            value -= self.learning_rate * value_grad


class LazyOptimizer:
    def __init__(self):
        pass
    
    def optimize(self, values, values_grad):
        pass
