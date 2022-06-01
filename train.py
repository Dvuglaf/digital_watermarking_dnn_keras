from tensorflow import keras
import tensorflow as tf
import numpy as np
import os


os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


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
    images = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/vov4i/Desktop/dataset",
                                                                 image_size=(128, 128),
                                                                 shuffle=False, batch_size=1)
    dataset = []
    for element in images.as_numpy_iterator():
        dataset.append(element[0] / 255)

    dataset = np.array(dataset)
    dataset = dataset.reshape((5200, 128, 128, 3))  # TODO: len(images)???
    dataset = dataset[:-5200+number]
    return dataset


def main():
    network = keras.models.load_model('full')
    watermark_dataset = get_watermark_dataset(1800)
    print("watermark dataset get successfully")
    coverimage_dataset = get_coverimage_dataset(1800)
    print("coverimage dataset get successfully")
    print("training...")
    # Model training
    history = network.fit([watermark_dataset, coverimage_dataset], [watermark_dataset, coverimage_dataset],
                          epochs=3,
                          batch_size=1,
                          validation_split=0.2,
                          shuffle=True
                          )
    network.save('full')
    print("FULL SAVED!")


main()

#TODO: Сделать этот модуль для обучения