import ssl
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# permission to download dataset from internet
ssl._create_default_https_context = ssl._create_unverified_context

# load a pre-defined dataset
fashion_mnist = keras.datasets.fashion_mnist.load_data()

# pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist