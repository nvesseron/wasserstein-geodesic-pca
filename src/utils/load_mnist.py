import array
import gzip
import os
import struct
import urllib.request
from os import path

import jax.numpy as jnp
import numpy as np


MNIST_DIR = path.join(os.environ.get("GPCAGEN_DATA_DIR", "NONE"), "mnist")
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _download(filename):
    os.makedirs(MNIST_DIR, exist_ok=True)
    out_file = path.join(MNIST_DIR, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(MNIST_URL + filename, out_file)
        print(f"downloaded {filename} to {MNIST_DIR}")


def _one_hot(labels, num_classes, dtype=np.float32):
    return np.array(labels[:, None] == np.arange(num_classes), dtype)


def _parse_labels(filename):
    with gzip.open(filename, "rb") as file:
        _ = struct.unpack(">II", file.read(8))
        return np.array(array.array("B", file.read()), dtype=np.uint8)


def _parse_images(filename):
    with gzip.open(filename, "rb") as file:
        _, num_data, rows, cols = struct.unpack(">IIII", file.read(16))
        return np.array(array.array("B", file.read()), dtype=np.uint8).reshape(num_data, rows, cols)


def mnist_raw():
    """Download missing IDX files and return raw MNIST arrays."""
    for filename in MNIST_FILES:
        _download(filename)

    train_images = _parse_images(path.join(MNIST_DIR, "train-images-idx3-ubyte.gz"))
    train_labels = _parse_labels(path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz"))
    test_images = _parse_images(path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz"))
    test_labels = _parse_labels(path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz"))
    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Return normalized MNIST images with a final channel axis and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()
    train_images = train_images / np.float32(255.0)
    test_images = test_images / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        permutation = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

    return (
        jnp.expand_dims(train_images, axis=-1),
        jnp.array(train_labels),
        jnp.expand_dims(test_images, axis=-1),
        jnp.array(test_labels),
    )
