import os
import cv2
from tensorflow import keras
import numpy
import tensorflow as tf
import tensorflow
import keras.backend as K
import numpy as np
import sklearn.model_selection
import copy
import pydot
import graphviz
from matplotlib import pyplot as plt
from skimage.io import imshow, show


model = keras.models.load_model('NETWORK_WITH_CUSTOM_LOSS.h5', compile=True)

train = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/aizee/Desktop/dataset", image_size=(128, 128), shuffle=False, batch_size=1)

res = []
for element in train.as_numpy_iterator():
    scale = element[0] / 255
    res.append(np.array(scale))
    if len(res) == 1920:
        break

res = np.array(res)
res = res.reshape((1920, 128, 128, 3))


def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = rgb2gray(x_train)
x_test = rgb2gray(x_test)
x_train = x_train[:-48080]
y_train = y_train[:-48080]
x_test = x_test[:-9520]
y_test = y_test[:-9520]

from skimage.io import imread

watermark  = x_test[0]
#coverimage = imread("C:/Users/aizee/Pictures/x.jpg").astype(np.float64)
#coverimage /= 255
coverimage = res[0]
data = [np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage).reshape((1, 128, 128, 3))]

print(data[0].shape)
print(data[1].shape)
res1, res2 = model.predict(data, batch_size=1)
fig = plt.figure()
sb = fig.add_subplot(2, 3, 1)
sb.set_title("Исходное изображение")
imshow(coverimage)

sb = fig.add_subplot(2, 3, 2)
sb.set_title("ЦВЗ")
imshow(watermark)

sb = fig.add_subplot(2, 3, 3)
sb.set_title("Изображение с ЦВЗ")
imshow(res2.reshape((128, 128, 3)))

sb = fig.add_subplot(2, 3, 4)
sb.set_title("Извлеченный ЦВЗ")
imshow(res1.reshape((32, 32, 1)), cmap='gray')

sb = fig.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - res1.reshape((32, 32)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ", np.max(np.max(coverimage - res2)))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ", np.max(np.abs(watermark - res1.reshape((32, 32)))))
show()