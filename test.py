from tensorflow import keras
import numpy
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow, show
import os
from skimage.io import imread, imsave
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#embedder = keras.models.load_model('embedder', compile=False)
#extractor = keras.models.load_model('extractor', compile=False)

network = keras.models.load_model('full')

train = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/vov4i/Desktop/dataset", image_size=(128, 128), shuffle=True, batch_size=1)
res = []
for element in train.as_numpy_iterator():
    res.append(element[0]/255)
    if len(res) == 1000:
        break

res = np.array(res)
res = res.reshape((1000, 128, 128, 3))

def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = rgb2gray(x_train)
x_test = rgb2gray(x_test)
x_train = x_train[:-49000]
y_train = y_train[:-48080]
x_test = x_test[:-9520]
y_test = y_test[:-9520]


watermark = x_train[16
]
print(watermark.shape)
from skimage.io import imread

coverimage_1 = imread("C:/Users/vov4i/Desktop/test_image.jpg").astype('float32') / 255
coverimage_2 = res[1]
coverimage_3 = res[999]
coverimage_4 = res[200]
coverimage_5 = res[639]

extracted_1, marked_1 = network.predict(
    [np.array(watermark).reshape((1, 32, 32)), np.array(coverimage_1).reshape((1, 128, 128, 3))], batch_size=1)
extracted_2, marked_2 = network.predict(
    [np.array(watermark).reshape((1,32, 32, 1)), np.array(coverimage_2).reshape((1, 128, 128, 3))], batch_size=1)
extracted_3, marked_3 = network.predict(
    [np.array(watermark).reshape((1,32, 32, 1)), np.array(coverimage_3).reshape((1, 128, 128, 3))], batch_size=1)
extracted_4, marked_4 = network.predict(
    [np.array(watermark).reshape((1,32, 32, 1)), np.array(coverimage_4).reshape((1, 128, 128, 3))], batch_size=1)
extracted_5, marked_5 = network.predict(
    [np.array(watermark).reshape((1,32, 32, 1)), np.array(coverimage_5).reshape((1, 128, 128, 3))], batch_size=1)

fig_1 = plt.figure()
sb = fig_1.add_subplot(2, 3, 1)
sb.set_title("Исходное изображение")
imshow(coverimage_1)

sb = fig_1.add_subplot(2, 3, 2)
sb.set_title("ЦВЗ")
imshow(watermark)

sb = fig_1.add_subplot(2, 3, 3)
sb.set_title("Изображение с ЦВЗ")
imshow(marked_1.reshape((128, 128, 3)))

sb = fig_1.add_subplot(2, 3, 4)
sb.set_title("Извлеченный ЦВЗ")
imshow(extracted_1.reshape((32, 32, 1)), cmap='gray')

sb = fig_1.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_1.reshape((32, 32)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ",
      np.max(np.abs(coverimage_1 - marked_1.reshape((128, 128, 3)))))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
      np.max(np.abs(watermark - extracted_1.reshape((32, 32)))))

fig_2 = plt.figure()
sb = fig_2.add_subplot(2, 3, 1)
sb.set_title("Исходное изображение")
imshow(coverimage_2)

sb = fig_2.add_subplot(2, 3, 2)
sb.set_title("ЦВЗ")
imshow(watermark)

sb = fig_2.add_subplot(2, 3, 3)
sb.set_title("Изображение с ЦВЗ")
imshow(marked_2.reshape((128, 128, 3)))

sb = fig_2.add_subplot(2, 3, 4)
sb.set_title("Извлеченный ЦВЗ")
imshow(extracted_2.reshape((32, 32)), cmap='gray')

sb = fig_2.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_2.reshape((32, 32)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ",
      np.max(np.abs(coverimage_2 - marked_2.reshape((128, 128, 3)))))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
      np.max(np.abs(watermark - extracted_2.reshape((32, 32)))))

fig_3 = plt.figure()
sb = fig_3.add_subplot(2, 3, 1)
sb.set_title("Исходное изображение")
imshow(coverimage_3)

sb = fig_3.add_subplot(2, 3, 2)
sb.set_title("ЦВЗ")
imshow(watermark)

sb = fig_3.add_subplot(2, 3, 3)
sb.set_title("Изображение с ЦВЗ")
imshow(marked_3.reshape((128, 128, 3)))

sb = fig_3.add_subplot(2, 3, 4)
sb.set_title("Извлеченный ЦВЗ")
imshow(extracted_3.reshape((32, 32)), cmap='gray')

sb = fig_3.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_3.reshape((32, 32)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ",
      np.max(np.abs(coverimage_3 - marked_3.reshape((128, 128, 3)))))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
      np.max(np.abs(watermark - extracted_3.reshape((32, 32)))))

fig_4 = plt.figure()
sb = fig_4.add_subplot(2, 3, 1)
sb.set_title("Исходное изображение")
imshow(coverimage_4)

sb = fig_4.add_subplot(2, 3, 2)
sb.set_title("ЦВЗ")
imshow(watermark)

sb = fig_4.add_subplot(2, 3, 3)
sb.set_title("Изображение с ЦВЗ")
imshow(marked_4.reshape((128, 128, 3)))

sb = fig_4.add_subplot(2, 3, 4)
sb.set_title("Извлеченный ЦВЗ")
imshow(extracted_4.reshape((32, 32)), cmap='gray')

sb = fig_4.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_4.reshape((32, 32)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ",
      np.max(np.abs(coverimage_4 - marked_4.reshape((128, 128, 3)))))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
      np.max(np.abs(watermark - extracted_4.reshape((32, 32)))))

fig_5 = plt.figure()
sb = fig_5.add_subplot(2, 3, 1)
sb.set_title("Исходное изображение")
imshow(coverimage_5)

sb = fig_5.add_subplot(2, 3, 2)
sb.set_title("ЦВЗ")
imshow(watermark)

sb = fig_5.add_subplot(2, 3, 3)
sb.set_title("Изображение с ЦВЗ")
imshow(marked_5.reshape((128, 128, 3)))

sb = fig_5.add_subplot(2, 3, 4)
sb.set_title("Извлеченный ЦВЗ")
imshow(extracted_5.reshape((32, 32)), cmap='gray')

sb = fig_5.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_5.reshape((32, 32)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ",
      np.max(np.abs(coverimage_5 - marked_5.reshape((128, 128, 3)))))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
      np.max(np.abs(watermark - extracted_5.reshape((32, 32)))))


show()

exit(111)
