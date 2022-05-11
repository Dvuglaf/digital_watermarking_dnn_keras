import copy
import os
from tensorflow import keras
import numpy
import tensorflow as tf
import tensorflow
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow, show

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    gram = tf.matmul(a, a, transpose_a=True)
    return gram


def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def conv_block_b(input_tensor):
    top = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
        input_tensor)

    mid = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
        input_tensor)
    mid = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=mid.shape[1:])(mid)

    bot = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
        input_tensor)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)

    concat = keras.layers.Concatenate(axis=len(input_tensor.shape) - 1)([top, mid, bot])
    concat = keras.layers.Conv2D(input_tensor.shape[3], (1, 1), padding='same', activation='relu',
                                 input_shape=input_tensor.shape[1:])(concat)
    res = concat + input_tensor
    return res


def conv_block_b_star(input_tensor):
    top = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
        input_tensor)

    mid = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
        input_tensor)
    mid = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=mid.shape[1:])(mid)

    bot = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
        input_tensor)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)

    concat = keras.layers.Concatenate(axis=len(input_tensor.shape) - 1)([top, mid, bot])

    b1 = concat

    b2 = keras.layers.Conv2D(input_tensor.shape[3], (1, 1), padding='same', activation='relu',
                              input_shape=input_tensor.shape[1:])(b1)

    res = b2 + input_tensor
    return res, b1, b2


# Inputs
watermark = keras.layers.Input(shape=[32, 32, 1])
coverimage = keras.layers.Input(shape=[128, 128, 3])

# Encoder
x = conv_block_b(watermark)
x = conv_block_b(x)
x = keras.layers.Conv2D(24, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
x = conv_block_b(x)
x = conv_block_b(x)
x = keras.layers.Conv2D(48, (1, 1), padding='same', name='1', activation='relu', input_shape=x.shape[1:])(x)
encoder_output = tensorflow.reshape(x, [-1, 128, 128, 3])

# Embedder
x, b1w, b2w = conv_block_b_star(encoder_output)
x = keras.layers.Concatenate(axis=len(x.shape) - 1)([x, coverimage])
x = conv_block_b(x)
embedder_output = keras.layers.Conv2D(3, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)

# Invariance network
N = 5
invariance_output = keras.layers.Dense(N, activation='relu', name='3')(embedder_output)
#add = conv_block_b_star(embedder_output)

# Extractor
x = conv_block_b(invariance_output)
x = conv_block_b(x)
extractor_output = keras.layers.Conv2D(3, (1, 1), padding='same', name='4', activation='relu', input_shape=x.shape[1:])(
    x)

# Decoder
x = tensorflow.reshape(extractor_output, [-1, 32, 32, 48])
x = conv_block_b(x)
x = conv_block_b(x)
x = keras.layers.Conv2D(24, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
x = conv_block_b(x)
x = conv_block_b(x)
decoder_output = keras.layers.Conv2D(1, (1, 1), padding='same', name='5', activation='relu', input_shape=x.shape[1:])(x)
f, b1m, b2m = conv_block_b_star(embedder_output)

# Single network model
network = keras.Model([watermark, coverimage], [decoder_output, embedder_output], name="network")

tf.keras.utils.plot_model(
    network,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True
)


# Loss block
# m_i       - marked image
# c_i       - coverimage
# w_star_i  - extracted and decoded watermark image
# w_i       - watermark image
def loss(w, c, w_f, m_i, b_1_m, b_1_w, b_2_m, b_2_w):
    loss_1 = keras.backend.mean(keras.backend.abs(w_f - w))
    loss_2 = keras.backend.mean(keras.backend.abs(m_i - c))

    loss_3 = 0.5 * (
            keras.backend.mean(keras.backend.abs(gram_matrix(b_1_w) - gram_matrix(b_1_m))) +
            keras.backend.mean(keras.backend.abs(gram_matrix(b_2_w) - gram_matrix(b_2_m)))
    )
    W = K.variable(value=network.get_layer('3').get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    h = network.get_layer('3').output
    dh = 1 - h * h  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = 0.01 * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

    return loss_1 + loss_2 + loss_3 + contractive


'''def loss(m_i, c_i, w_star_i, w_i, int_w_1, int_w_2, int_m_1, int_m_2):
    loss_1 = tf.keras.backend.mean(tf.keras.backend.abs(w_star_i - w_i))
    print(loss_1.shape)
    loss_2 = tf.keras.backend.mean(tf.keras.backend.abs(m_i - c_i))
    print(loss_2.shape)
    loss_3 = 0.5 * (
                    tf.keras.backend.mean(tf.keras.backend.abs(gram_matrix(int_w_1) - gram_matrix(int_m_1))) +
                    tf.keras.backend.mean(tf.keras.backend.abs(gram_matrix(int_w_2) - gram_matrix(int_m_2)))
                    )
    W = K.variable(value=network.get_layer('3').get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    h = network.get_layer('3').output
    dh = 1 - h*h  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = 0.01 * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)
    print(contractive.shape)

    return loss_1 + loss_2 + contractive + loss_3#+ loss_3 + contractive
'''

#network.add_loss(loss(watermark, coverimage, decoder_output, embedder_output, b1m, b1w, b2m, b2w))

network.compile(optimizer='adam', loss='mae')
# Training datasets
#   Watermark (cifar10, 32x32x1)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = rgb2gray(x_train)
x_test = rgb2gray(x_test)
x_train = x_train[:-49000]  # to 1920 images
y_train = y_train[:-48080]
x_test = x_test[:-9520]
y_test = y_test[:-9520]
x_train = x_train.reshape((1000, 32, 32, 1))
print(x_train.shape)

#   Coverimage (dataset from internet, 128x128x3)
# TODO: wrap into function
train = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/aizee/OneDrive/Desktop/dataset", image_size=(128, 128),
                                                            shuffle=True, batch_size=1)
c_train = []
for element in train.as_numpy_iterator():
    c_train.append(element[0] / 255)

c_train = np.array(c_train)
c_train = c_train.reshape((4400, 128, 128, 3))
c_test = c_train[1920:3840]
res = c_train[:-3400]
print(res.shape)

# Model training
history = network.fit([np.array(x_train), np.array(res)], [np.array(x_train), np.array(res)],
                      epochs=5,
                      batch_size=1,
                      validation_split=0.2,
                      shuffle=True
                      )
network.save('full')
print('\nhistory dict:', history.history)
# print('\n# Оцениваем на тестовых данных')
# results = network.evaluate([x_test, c_test], [x_test, c_test], batch_size=3)
# print('test loss, test acc:', results)
# Set weights from full network to submodels (encoder + embedder & invariance + extractor + decoder)
embedder = keras.Model(inputs=[watermark, coverimage], outputs=[embedder_output], name="embedder")
embedder.compile(optimizer="adam")
extractor = keras.Model(inputs=[embedder_output], outputs=[decoder_output], name="extractor")
extractor.compile(optimizer="adam")

for i in range(0, len(embedder.layers), 1):
    embedder.layers[i].set_weights(network.layers[i].get_weights())
for i in range(len(embedder.layers) + 1, len(embedder.layers) + len(extractor.layers)):
    extractor.layers[i - len(embedder.layers)].set_weights(network.layers[i - 1].get_weights())
network.save('full')
print("SAVED!")

watermark = x_train[0]
from skimage.io import imread

coverimage_1 = imread("C:/Users/aizee/Pictures/x.jpg").astype('float32') / 255
coverimage_2 = c_train[1]
coverimage_3 = c_train[1000]
coverimage_4 = c_train[3000]
coverimage_5 = c_train[4000]

extracted_1, marked_1 = network.predict(
    [np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage_1).reshape((1, 128, 128, 3))], batch_size=1)
extracted_2, marked_2 = network.predict(
    [np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage_2).reshape((1, 128, 128, 3))], batch_size=1)
extracted_3, marked_3 = network.predict(
    [np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage_3).reshape((1, 128, 128, 3))], batch_size=1)
extracted_4, marked_4 = network.predict(
    [np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage_4).reshape((1, 128, 128, 3))], batch_size=1)
extracted_5, marked_5 = network.predict(
    [np.array(watermark).reshape((1, 32, 32, 1)), np.array(coverimage_5).reshape((1, 128, 128, 3))], batch_size=1)

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
imshow(watermark - extracted_1.reshape((32, 32, 1)), cmap='gray')

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
imshow(extracted_2.reshape((32, 32, 1)), cmap='gray')

sb = fig_2.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_2.reshape((32, 32, 1)), cmap='gray')

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
imshow(extracted_3.reshape((32, 32, 1)), cmap='gray')

sb = fig_3.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_3.reshape((32, 32, 1)), cmap='gray')

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
imshow(extracted_4.reshape((32, 32, 1)), cmap='gray')

sb = fig_4.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_4.reshape((32, 32, 1)), cmap='gray')

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
imshow(extracted_5.reshape((32, 32, 1)), cmap='gray')

sb = fig_5.add_subplot(2, 3, 5)
sb.set_title("Разница между ЦВЗ и извлеч. ЦВЗ")
imshow(watermark - extracted_5.reshape((32, 32, 1)), cmap='gray')

print("Максимальная разница м/у исходным и изобр. с ЦВЗ",
      np.max(np.abs(coverimage_5 - marked_5.reshape((128, 128, 3)))))
print("Максимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
      np.max(np.abs(watermark - extracted_5.reshape((32, 32)))))

embedder.save("embedder")
extractor.save("extractor")

show()

exit(111)

# test_scores = network.evaluate([x_train, res], [x_train, res], batch_size=1, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
