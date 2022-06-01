import random

from tensorflow import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow, show
import utils


def create_model():
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
        b2 = keras.layers.Conv2D(input_tensor.shape[3], (1, 1), padding='same', activation='relu',
                                 input_shape=input_tensor.shape[1:])(concat)
        res = b2 + input_tensor
        return res, concat, b2
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
    encoder_output = tf.reshape(x, [-1, 128, 128, 3])

    # Embedder
    x, b1w, b2w = conv_block_b_star(encoder_output)
    x = keras.layers.Concatenate(axis=len(x.shape) - 1)([x, coverimage])
    x = conv_block_b(x)
    marked = keras.layers.Conv2D(3, (1, 1), padding='same', name='2', activation='sigmoid', input_shape=x.shape[1:])(x)
    embedder_output_1 = keras.layers.Conv2D(3, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
    embedder_output_2 = embedder_output_1

    # Invariance network
    N = 5
    invariance_output = keras.layers.Dense(N, activation='relu', name='3')(embedder_output_1)

    # Extractor
    x = conv_block_b(invariance_output)
    x = conv_block_b(x)
    extractor_output = keras.layers.Conv2D(3, (1, 1), padding='same', name='4', activation='relu',
                                           input_shape=x.shape[1:])(x)

    # Decoder
    x = tf.reshape(extractor_output, [-1, 32, 32, 48])
    x = conv_block_b(x)
    x = conv_block_b(x)
    x = keras.layers.Conv2D(24, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
    x = conv_block_b(x)
    x = conv_block_b(x)
    decoder_output = keras.layers.Conv2D(1, (1, 1), padding='same', name='5', activation='sigmoid',
                                         input_shape=x.shape[1:])(x)

    f, b1m, b2m = conv_block_b_star(embedder_output_2)

    # Single network model
    network = keras.Model([watermark, coverimage], [decoder_output, marked], name="network")

    # Loss block
    # w_i           - watermark image
    # c_i           - coverimage
    # w_star_i      - extracted and decoded watermark image
    # m_i           - marked image
    # b_1_m, b_2_w   - feature space of marked image
    # b_1_w, b_2_w   - feature space of encoded watermark
    def loss(w_i, c_i, w_star_i, m_i, b_1_m, b_1_w, b_2_m, b_2_w):
        loss_1 = keras.backend.mean(keras.backend.abs(w_star_i - w_i))
        loss_2 = keras.backend.mean(keras.backend.abs(m_i - c_i))
        loss_3 = 0.5 * (
                keras.backend.mean(keras.backend.abs(utils.gram_matrix(b_1_w) - utils.gram_matrix(b_1_m))) +
                keras.backend.mean(keras.backend.abs(utils.gram_matrix(b_2_w) - utils.gram_matrix(b_2_m)))
        )
        W = keras.backend.variable(value=network.get_layer('3').get_weights()[0])  # N x N_hidden
        W = keras.backend.transpose(W)  # N_hidden x N
        h = network.get_layer('3').output
        dh = 1 - h * h  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = 0.01 * keras.backend.sum(dh ** 2 * keras.backend.sum(W ** 2, axis=1), axis=2)

        return loss_1 + loss_2 + loss_3 + contractive

    network.add_loss(loss(watermark, coverimage, decoder_output, marked, b1m, b1w, b2m, b2w))
    network.compile(optimizer='adam')
    print("network built successfully")
    return network


def train_model(model, num_epochs, batch_size, path_to_save):
    watermark_dataset = utils.get_watermark_dataset(1800)
    print("watermark dataset get successfully")
    coverimage_dataset = utils.get_coverimage_dataset(1800)
    print("coverimage dataset get successfully")

    # Model training
    print("training...")
    history = model.fit([watermark_dataset, coverimage_dataset], [watermark_dataset, coverimage_dataset],
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        shuffle=True
                        )
    print("training ends")
    print('history dict:', history.history)

    # Save model
    model.save(path_to_save)
    print("saved!")


def predict_model(path_to_model, num_predicted):
    network = keras.models.load_model(path_to_model)

    watermark_dataset = utils.get_watermark_dataset(1800)
    coverimage_dataset = utils.get_coverimage_dataset(1800)

    for i in range(num_predicted):
        watermark = watermark_dataset[random.randint(0, 1800)]
        coverimage = coverimage_dataset[random.randint(0, 1800)]

        extracted, marked = network.predict(
            [np.array(watermark).reshape((1, 32, 32)), np.array(coverimage).reshape((1, 128, 128, 3))], batch_size=1)

        fig = plt.figure()
        sb = fig.add_subplot(2, 2, 1)
        sb.set_title("Исходное изображение")
        imshow(coverimage)

        sb = fig.add_subplot(2, 2, 2)
        sb.set_title("ЦВЗ")
        imshow(watermark)

        sb = fig.add_subplot(2, 2, 3)
        sb.set_title("Изображение с ЦВЗ")
        imshow(marked.reshape((128, 128, 3)))

        sb = fig.add_subplot(2, 2, 4)
        sb.set_title("Извлеченный ЦВЗ")
        imshow(extracted.reshape((32, 32, 1)), cmap='gray')

        print(str(i) + ":")
        print("\tМаксимальная разница м/у исходным и изобр. с ЦВЗ",
              np.max(np.abs(coverimage - marked.reshape((128, 128, 3)))))
        print("\tМаксимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
              np.max(np.abs(watermark - extracted.reshape((32, 32)))))
        print("\tСредняя разница м/у исходным и изобр. с ЦВЗ",
              np.mean(np.abs(coverimage - marked.reshape((128, 128, 3)))))
        print("\tСредняя разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
              np.mean(np.abs(watermark - extracted.reshape((32, 32)))))
    show()
