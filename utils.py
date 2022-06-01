import os
import tensorflow as tf
from tensorflow import keras
import numpy as np


def init():
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def gram_matrix(arr):
    channels = int(arr.shape[-1])
    a = tf.reshape(arr, [-1, channels])
    gram = tf.matmul(a, a, transpose_a=True)
    return gram


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# Watermark: from CIFAR10 dataset (number, 32, 32, 1) shape
def get_watermark_dataset(number):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    dataset = x_train.astype('float32') / 255.
    dataset = rgb2gray(dataset)
    dataset = dataset[:-50000+number]  # to number images
    dataset = dataset.reshape((number, 32, 32, 1))
    return dataset


# Coverimage: from ImageNET dataset (number, 128, 128, 3) shape
def get_coverimage_dataset(number):
    images = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/aizee/OneDrive/Desktop/dataset/train",
                                                                 image_size=(128, 128),
                                                                 shuffle=False, batch_size=1)
    dataset = []
    for element in images.as_numpy_iterator():
        dataset.append(element[0] / 255)

    dataset = np.array(dataset)
    dataset = dataset.reshape((2400, 128, 128, 3))  # TODO: len(images)???
    dataset = dataset[:-2400+number]
    return dataset


# Set weights from full network to submodels (encoder + embedder & invariance + extractor + decoder)
def create_submodels(model):
    embedder = keras.Model(inputs=[model.inputs[0], model.inputs[1]], outputs=[model.outputs[1]], name="embedder")
    embedder.compile(optimizer="adam")
    extractor = keras.Model(inputs=[model.outputs[1]], outputs=[model.outputs[0]], name="extractor")
    extractor.compile(optimizer="adam")

    for i in range(0, len(embedder.layers), 1):
        embedder.layers[i].set_weights(model.layers[i].get_weights())
    for i in range(len(embedder.layers) + 1, len(embedder.layers) + len(extractor.layers)):
        extractor.layers[i - len(embedder.layers)].set_weights(model.layers[i - 1].get_weights())

    embedder.save("embedder")
    extractor.save("extractor")
    print("embedder saved!")
    print("extractor saved!")
