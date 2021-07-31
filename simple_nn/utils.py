import numpy as np


def one_hot_digit(labels):
    eye = np.eye(10)
    return eye[labels]
