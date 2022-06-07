import tensorflow as tf
from tensorflow import keras
import numpy as np


# Return Gram matrix (all possible dot products)
def gram_matrix(mat):
    channels = int(mat.shape[-1])
    a = tf.reshape(mat, [-1, channels])
    gram = tf.matmul(a, a, transpose_a=True)
    return gram


# Convert RGB to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# Get Watermark dataset from CIFAR10 dataset, shape=(number, 32, 32, 1)
def get_watermark_dataset(number):
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

    dataset = x_train.astype('float32') / 255.
    dataset = rgb2gray(dataset)
    dataset = dataset[:-50000+number]  # to number images
    dataset = dataset.reshape((number, 32, 32, 1))
    return np.array(dataset)


# Get Coverimage dataset from ImageNET dataset, shape=(number, 128, 128, 3)
def get_coverimage_dataset(number, path_to_dataset):
    images = tf.keras.preprocessing.image_dataset_from_directory(path_to_dataset, image_size=(128, 128),
                                                                 shuffle=True, batch_size=1
                                                                 )
    dataset = []
    for element in images.as_numpy_iterator():
        dataset.append(element[0] / 255)

    dataset = np.array(dataset)
    dataset = dataset.reshape((len(images), 128, 128, 3))
    dataset = dataset[:-len(images)+number]
    return dataset
