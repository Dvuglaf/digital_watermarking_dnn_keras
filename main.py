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

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


def get_img(data_path):
    # Getting image array from path:
    img = cv2.imread(data_path, cv2.IMREAD_COLOR).astype('float32') / 255
    return img


def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_dataset(dataset_path='C:/Users/aizee/Desktop/dataset/train'):
    num_class = 2
    test_size = 0.2
    # Getting all data from data path:
    try:
        X = np.load('npy_dataset/X.npy')
        Y = np.load('npy_dataset/Y.npy')
    except:
        labels = os.listdir(dataset_path)  # Getting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path + '/' + label
            for data in os.listdir(datas_path):
                img = get_img(datas_path + '/' + data)
                X.append(img)
                Y.append(i)
        # Create dateset:
        X = 1 - np.array(X).astype('float32') / 255.
        Y = np.array(Y).astype('float32')
        Y = keras.utils.to_categorical(Y, num_class)
        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
    X, X_test, Y, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, random_state=42)
    return X, X_test, Y, Y_test


def rgb2gray(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def conv_block_b(input_tensor):

    top = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(input_tensor)

    mid = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(input_tensor)
    mid = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=mid.shape[1:])(mid)

    bot = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(input_tensor)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)

    concat = keras.layers.Concatenate(axis=len(input_tensor.shape) - 1)([top, mid, bot])
    concat = keras.layers.Conv2D(input_tensor.shape[3], (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(concat)
    res = concat + input_tensor
    return res


def conv_block_b_star(input_tensor):
    tf.compat.v1.keras.layers.BatchNormalization()
    top = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(input_tensor)
    tf.compat.v1.keras.layers.BatchNormalization(axis=1)

    mid = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(input_tensor)
    tf.compat.v1.keras.layers.BatchNormalization(axis=1)

    mid = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=mid.shape[1:])(mid)
    tf.compat.v1.keras.layers.BatchNormalization(axis=1)

    bot = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(input_tensor)
    tf.compat.v1.keras.layers.BatchNormalization(axis=1)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)
    tf.compat.v1.keras.layers.BatchNormalization(axis=1)
    bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)
    concat = keras.layers.Concatenate(axis=len(input_tensor.shape) - 1)([top, mid, bot])
    tf.compat.v1.keras.layers.BatchNormalization()
    b_1 = tf.identity(concat)
    concat = keras.layers.Conv2D(input_tensor.shape[3], (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(concat)
    tf.compat.v1.keras.layers.BatchNormalization(axis=1)
    b_2 = tf.identity(concat)
    res = concat + input_tensor
    return res, b_1, b_2


def cust(tensor):
    return tf.norm(tensor)


watermark = keras.layers.Input(shape=[32, 32, 1])  # Encoder input
coverimage = keras.layers.Input(shape=[128, 128, 3])

# Encoder
x = conv_block_b(watermark)
x = conv_block_b(x)
x = keras.layers.Conv2D(24, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
x = conv_block_b(x)
x = conv_block_b(x)
x = keras.layers.Conv2D(48, (1, 1), padding='same', name='1', activation='relu', input_shape=x.shape[1:])(x)
encoder_output = tensorflow.reshape(x, [-1, 128, 128, 3])
#encoder = keras.Model(inputs=[watermark, coverimage], outputs=[encoder_output], name='encoder')

# Embedder
#   inputs  = [encoder_output, coverimage_input]
#   outputs = [m_i, b_1, b_2]

#embedder_input_1 = keras.layers.Input(shape=[128, 128, 3])
#embedder_input_2 = keras.layers.Input(shape=[128, 128, 3], name="coverimage")
x, b_1_w, b_2_w = conv_block_b_star(encoder_output)
f, b_1_m, b_2_m = conv_block_b_star(coverimage)
x = keras.layers.Concatenate(axis=len(x.shape) - 1)([x, coverimage])
x = conv_block_b(x)
embedder_output = keras.layers.Conv2D(3, (1, 1), name='2', padding='same', activation='relu', input_shape=x.shape[1:])(x)

#embedder = keras.Model(inputs=[coverimage], outputs=[embedder_output_1, b_1_w, b_2_w, b_1_m, b_2_m], name='embedder')

# Invariance layer
#   inputs  = [m_i]
#   outputs = [t_i]
#invariance_input = keras.layers.Input(shape=[128, 128, 3])
N = 5
invariance_output = keras.layers.Dense(5, activation='relu', name='3')(embedder_output)
#invariance = keras.Model(inputs=[embedder_output], outputs=[invariance_output], name='invariance')

# Extractor
#   inputs  = [t_i]
#   outputs = [w_f_star]
#extractor_input = keras.layers.Input(shape=[128, 128, N])
x = conv_block_b(invariance_output)
x = conv_block_b(x)
extractor_output = keras.layers.Conv2D(3, (1, 1), padding='same', name='4', activation='relu', input_shape=x.shape[1:])(x)

#extractor = keras.Model(inputs=[invariance.output], outputs=[extractor_output], name='extractor')

# Decoder
#   inputs  = [w_f_star]
#   outputs = [w_i_star]
#decoder_input = keras.layers.Input(shape=[128, 128, 3])
x = tensorflow.reshape(extractor_output, [-1, 32, 32, 48])
x = conv_block_b(x)
x = conv_block_b(x)
x = keras.layers.Conv2D(24, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
x = conv_block_b(x)
x = conv_block_b(x)
decoder_output = keras.layers.Conv2D(1, (1, 1), padding='same', name='5', activation='relu', input_shape=x.shape[1:])(x)

#decoder = keras.Model(inputs=[extractor.output], outputs=[decoder_output], name='decoder')

#watermark_input = keras.layers.Input(shape=[32, 32, 1])
#coverimage_input = keras.layers.Input(shape=[128, 128, 3])

'''encoded_w = encoder(encoder_input)
embeddered_img, b_1_w, b_2_w, b_1_m, b_2_m = embedder([encoded_w, embedder_input_2])
feature_space_img = invariance(embeddered_img)
extracted_w = extractor(feature_space_img)
decoded_w = decoder(extracted_w)'''
network = keras.Model([watermark, coverimage], [decoder_output, embedder_output], name="network")


# Loss block
# m_i       - marked image
# c_i       - coverimage
# w_star_i  - extracted and decoded watermark image
# w_i       - watermark image


def loss(m_i, c_i, w_star_i, w_i, int_w_1, int_w_2, int_m_1, int_m_2):
    loss_1 = tf.reduce_mean(tf.abs(w_star_i - w_i))
    loss_2 = tf.reduce_mean(tf.abs(m_i - c_i))
    loss_3 = 0.5 * (tf.norm(gram_matrix(int_w_1) - gram_matrix(int_m_1)) +
                    tf.norm(gram_matrix(int_w_2) - gram_matrix(int_m_2)))

    W = K.variable(value=network.get_layer('3').get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    h = network.get_layer('3').output
    dh = 1 - h*h  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = 0.01 * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)
    return loss_1 + loss_2 + contractive


network.add_loss(loss(embedder_output, coverimage, decoder_output, watermark, b_1_w, b_2_w, b_1_m, b_2_m))
#network.add_loss(loss(tensorflow.reshape(embedder_output, [-1, 128, 128, 3]), coverimage, decoder_output, watermark,
#                      b_1_w, b_2_w, b_1_m, b_2_m))
#network.add_loss(tf.norm(decoded_w - watermark_input) + tf.norm(embeddered_img - coverimage_input))
#network.add_loss(tf.norm(w_i_star - watermark_input) + tf.norm(m_i - coverimage_input) + tf.norm(b_1) - tf.norm(b_2))
#network.add_loss(tf.norm(watermark_input - decoded_w, ord=1) + tf.norm(gram_matrix(b_1)) + tf.norm(gram_matrix(b_2)))

'''
tf.keras.utils.plot_model(
    network,
    to_file="network.png",
    show_shapes=True,
    show_dtype=True
) # plot_model
'''
network.compile(optimizer='adam')

# Watermark
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = rgb2gray(x_train)
x_test = rgb2gray(x_test)
x_train = x_train[:-48080]
y_train = y_train[:-48080]
x_test = x_test[:-9520]
y_test = y_test[:-9520]

print("Watermark shapes:")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Coverimage
train = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/aizee/Desktop/dataset", image_size=(128, 128), shuffle=False, batch_size=1)

res = []
for element in train.as_numpy_iterator():
    scale = element[0] / 255
    res.append(np.array(scale))
    if len(res) == 1920:
        break

res = np.array(res)
res = res.reshape((1920, 128, 128, 3))
print(res.shape)

history = network.fit([x_train, res], [x_train, res],
                      epochs=7,
                      batch_size=1,
                      validation_split=0.2,
                      shuffle=True
                      )
network.save('NETWORK_WITH_CUSTOM_LOSS.h5')


test_scores = network.evaluate([x_test, res], [x_test, res], batch_size=1, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
print(history.history)


# 1. Разобраться с функциями потерь!!!
# 2. Понять как использовать B1 и B2