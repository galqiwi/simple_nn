import numpy as np
from scipy.special import softmax
from abc import ABC, abstractmethod
from scipy.special import expit


class Layer(ABC):
    @abstractmethod
    def forward(self, input_v):
        pass

    @abstractmethod
    def backward(self, input_v, output_v, output_grad_v):
        pass

    @abstractmethod
    def get_gradients(self):
        pass

    @abstractmethod
    def get_values(self):
        pass


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.M = np.random.normal(scale=1 / (input_size ** 0.5), size=(output_size, input_size))  # M is for Matrix
        self.M_grad = None
        self.B = np.random.normal(scale=1, size=(output_size,))  # B is for bias
        self.B_grad = None

    def forward(self, input_v):
        return (self.M @ input_v[:, :, None])[:, :, 0] + self.B

    def backward(self, input_v, output_v, output_grad_v):
        self.M_grad = output_grad_v.T @ input_v
        self.B_grad = np.sum(output_grad_v, axis=0)
        return output_grad_v @ self.M

    def get_gradients(self):
        if self.M_grad is None:
            raise ValueError('no gradient value, run backprop before calling get_gradients()')
        return [self.M_grad, self.B_grad]

    def get_values(self):
        return [self.M, self.B]


class ReluLayer(Layer):
    def __init__(self):
        pass

    def forward(self, input_v):
        return input_v * (input_v > 0)

    def backward(self, input_v, output_v, output_grad_v):
        return output_grad_v * (input_v > 0)

    def get_gradients(self):
        return list()

    def get_values(self):
        return list()


class LeakyReluLayer(Layer):
    def __init__(self, leak=0.05):
        self.leak = leak
        pass

    def forward(self, input_v):
        return input_v * (input_v > 0) + input_v * (input_v <= 0) * self.leak

    def backward(self, input_v, output_v, output_grad_v):
        return output_grad_v * (input_v > 0) + output_grad_v * (input_v <= 0) * self.leak

    def get_gradients(self):
        return list()

    def get_values(self):
        return list()


class SoftMaxLayer(Layer):
    def __init__(self):
        pass

    def forward(self, input_v):
        return softmax(input_v, axis=-1)

    def backward(self, input_v, output_v, output_grad_v):
        grad_matrix = - output_v[:, None, :] * output_v[:, :, None] + np.eye(output_v.shape[-1]) * output_v[:, :, None]
        return (grad_matrix @ output_grad_v[:, :, None])[:, :, 0]

    def get_gradients(self):
        return list()

    def get_values(self):
        return list()


class SigmaLayer(Layer):
    def __init__(self):
        pass

    def forward(self, input_v):
        return expit(input_v)

    def backward(self, input_v, output_v, output_grad_v):
        return output_v * (1 - output_v) * output_grad_v

    def get_gradients(self):
        return list()

    def get_values(self):
        return list()
