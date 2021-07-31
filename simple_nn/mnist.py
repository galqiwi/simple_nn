import requests
import numpy as np
import gzip
import pickle
import os

files = [
    {
        'name': 'training_images',
        'filename': 'train-images-idx3-ubyte.gz',
        'offset': 16,
        'shape': (-1, 28 * 28)
    },
    {
        'name': 'test_images',
        'filename': 't10k-images-idx3-ubyte.gz',
        'offset': 16,
        'shape': (-1, 28 * 28)
    },
    {
        'name': 'training_labels',
        'filename': 'train-labels-idx1-ubyte.gz',
        'offset': 8,
        'shape': (-1,)
    },
    {
        'name': 'test_labels',
        'filename': 't10k-labels-idx1-ubyte.gz',
        'offset': 8,
        'shape': (-1,)
    }
]
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist"


def download_mnist(filepath='mnist.pickle'):
    mnist = dict()
    for file in files:
        print(f'Downloading {file["name"]}')
        data = requests.get(f'{base_url}/{file["filename"]}').content
        mnist[file['name']] = np.frombuffer(
            gzip.decompress(data), np.uint8, offset=file['offset']
        ).reshape(file['shape'])

    with open(filepath, 'wb') as file:
        pickle.dump(mnist, file)

    return mnist


def load_mnist(filepath='mnist.pickle'):
    if not os.path.isfile(filepath):
        return download_mnist(filepath)

    with open(filepath, 'rb') as file:
        return pickle.load(file)
