import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import pickle
# sns.set(font_scale=1.6)

import simple_nn.mnist as mnist
from simple_nn.layers import DenseLayer, ReluLayer, SoftMaxLayer, LeakyReluLayer, SigmaLayer
from simple_nn.loss_functions import MSE
from simple_nn.optimizers import SGD, LazyOptimizer
from simple_nn.model import Model
from simple_nn.utils import one_hot_digit

mnist_data = mnist.load_mnist()
train_images = mnist_data['training_images']
train_labels = mnist_data['training_labels']
test_images = mnist_data['test_images']
test_labels = mnist_data['test_labels']

train_size = train_labels.shape[0]
test_size = test_labels.shape[0]
train_images = train_images / 255.
test_images = test_images / 255.


class TestGradient(unittest.TestCase):
    def test_gradient(self):
        model = Model()
        model.add_layer(DenseLayer(input_size=28 * 28, output_size=10))
        model.add_layer(SoftMaxLayer())
        model.add_layer(DenseLayer(input_size=10, output_size=10))
        model.add_layer(LeakyReluLayer())
        model.add_layer(DenseLayer(input_size=10, output_size=10))
        model.add_layer(ReluLayer())
        model.add_layer(DenseLayer(input_size=10, output_size=10))
        model.add_layer(SigmaLayer())

        def get_error():
            return model.fit(train_images[:10], one_hot_digit(train_labels[:10]), MSE(), LazyOptimizer())

        error_base = get_error()

        for layer in model.layers:
            values = layer.get_values()
            values_grad = layer.get_gradients()

            for value, value_grad in zip(values, values_grad):
                d_value = np.random.normal(size=value.shape) / 1e5

                d_error_theory = np.sum(d_value * value_grad)

                value += d_value
                new_error = get_error()
                d_error_real = new_error - error_base
                error_base = new_error
                np.testing.assert_almost_equal(d_error_theory, d_error_real)


if __name__ == '__main__':
    unittest.main()
