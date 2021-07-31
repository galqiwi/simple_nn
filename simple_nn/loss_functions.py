from abc import ABC, abstractmethod
import numpy as np


class LossClass(ABC):
    @abstractmethod
    def forward(self, actual, desired):
        pass

    @abstractmethod
    def backward(self, actual, desired, error):
        pass


class MSE(LossClass):  # TODO: add abstract class
    def __init__(self):
        pass

    def forward(self, actual, desired):
        return np.mean((actual - desired) ** 2)

    def backward(self, actual, desired, error):
        return 2 * (actual - desired) / np.prod(actual.shape)


class SE(LossClass):  # TODO: add abstract class
    def __init__(self):
        pass

    def forward(self, actual, desired):
        return np.mean((actual - desired) ** 2, axis=-1)

    def backward(self, actual, desired, error):
        raise ValueError('SE does not support backprop')
