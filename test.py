from tensorflow import keras
import numpy
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow, show
from skimage.io import imread, imsave

#embedder = keras.models.load_model('embedder', compile=False)
#extractor = keras.models.load_model('extractor', compile=False)

network = keras.models.load_model('full')

train = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/aizee/Desktop/dataset", image_size=(128, 128), shuffle=True, batch_size=1)
res = []
for element in train.as_numpy_iterator():
    res.append(element[0]/255)
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


watermark  = x_train[0]
coverimage = res[0]

#imread("C:/Users/aizee/Pictures/x.jpg").astype('float32')/255
print(coverimage)
#coverimage = res[2]
#data = [np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage).reshape((1, 128, 128, 3))]

extracted, marked = network.predict([np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage).reshape((1, 128, 128, 3))], batch_size=1)
print(marked)
#extracted = extractor.predict(marked, batch_size=1)
fig = plt.figure()
sb = fig.add_subplot(2, 3, 1)
sb.set_title("Исходное изображение")
imshow(coverimage.reshape((128, 128, 3)))

sb = fig.add_subplot(2, 3, 2)
sb.set_title("ЦВЗ")
imshow(watermark.reshape((32, 32)), cmap='gray')

sb = fig.add_subplot(2, 3, 3)
sb.set_title("Изображение с ЦВЗ")
imshow(marked.reshape((128, 128, 3)), cmap = 'gray')

sb = fig.add_subplot(2, 3, 4)
sb.set_title("Извлеченный ЦВЗ")
imshow(extracted.reshape((32, 32, 1)), cmap='gray')

sb = fig.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted.reshape((32, 32)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ", np.max(np.abs(coverimage - marked.reshape((128, 128, 3)))))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ", np.max(np.abs(watermark - extracted.reshape((32, 32)))))
imsave("C:/Users/aizee/Pictures/marked.jpg", marked.reshape((128, 128, 3)))
show()