import random

from tensorflow import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow, show
import utils


# Construct keras Functional model of the proposed method of digital watermarking
def create_model():
    # Residual block B
    def conv_block_b(input_tensor):
        top = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
            input_tensor
        )

        mid = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
            input_tensor
        )
        mid = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=mid.shape[1:])(mid)

        bot = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
            input_tensor
        )
        bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)
        bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)

        concat = keras.layers.Concatenate(axis=len(input_tensor.shape) - 1)([top, mid, bot])
        conv_concat = keras.layers.Conv2D(input_tensor.shape[3],
                                          (1, 1),
                                          padding='same',
                                          activation='relu',
                                          input_shape=input_tensor.shape[1:]
                                          )(concat)

        return conv_concat + input_tensor

    # Residual block B that extract feature map from w_f and m
    def conv_block_b_star(input_tensor):
        top = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
            input_tensor
        )

        mid = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
            input_tensor
        )
        mid = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=mid.shape[1:])(mid)

        bot = keras.layers.Conv2D(32, (1, 1), padding='same', activation='relu', input_shape=input_tensor.shape[1:])(
            input_tensor
        )
        bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)
        bot = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=bot.shape[1:])(bot)

        b1 = keras.layers.Concatenate(axis=len(input_tensor.shape) - 1)([top, mid, bot])
        b2 = keras.layers.Conv2D(input_tensor.shape[3],
                                 (1, 1),
                                 padding='same',
                                 activation='relu',
                                 input_shape=input_tensor.shape[1:]
                                 )(b1)

        return b2 + input_tensor, b1, b2

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

    # Invariance network
    N = 5
    x = keras.layers.BatchNormalization(axis=3)(marked)
    invariance_output = keras.layers.Dense(N, activation='tanh', name='3')(x)

    # Extractor
    x = keras.layers.BatchNormalization(axis=3)(invariance_output)
    x = conv_block_b(x)
    x = conv_block_b(x)
    extractor_output = keras.layers.Conv2D(3,
                                           (1, 1),
                                           padding='same',
                                           name='4',
                                           activation='relu',
                                           input_shape=x.shape[1:]
                                           )(x)

    # Decoder
    x = tf.reshape(extractor_output, [-1, 32, 32, 48])
    x = conv_block_b(x)
    x = conv_block_b(x)
    x = keras.layers.Conv2D(24, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
    x = conv_block_b(x)
    x = conv_block_b(x)
    decoder_output = keras.layers.Conv2D(1,
                                         (1, 1),
                                         padding='same',
                                         name='5',
                                         activation='sigmoid',
                                         input_shape=x.shape[1:]
                                         )(x)

    f, b1m, b2m = conv_block_b_star(embedder_output_1)  # to extract feature map from marked

    # Single network model
    network = keras.Model([watermark, coverimage], [decoder_output, marked], name="network")

    # Loss block
    # w_i            - watermark image
    # c_i            - coverimage
    # w_star_i       - extracted and decoded watermark image
    # m_i            - marked image
    # b_1_m, b_2_w   - feature map of marked image
    # b_1_w, b_2_w   - feature map of encoded watermark
    def loss(w_i, c_i, w_star_i, m_i, b_1_m, b_1_w, b_2_m, b_2_w):
        loss_1 = keras.backend.mean(keras.backend.abs(w_star_i - w_i))

        loss_2 = keras.backend.mean(keras.backend.abs(m_i - c_i))

        loss_3 = 0.5 * (
                keras.backend.mean(keras.backend.abs(utils.gram_matrix(b_1_w) - utils.gram_matrix(b_1_m))) +
                keras.backend.mean(keras.backend.abs(utils.gram_matrix(b_2_w) - utils.gram_matrix(b_2_m)))
        )

        W = keras.backend.variable(value=network.get_layer('3').get_weights()[0])
        W = keras.backend.transpose(W)
        h = network.get_layer('3').output
        dh = 1 - h * h

        contractive = 0.01 * keras.backend.sum(dh ** 2 * keras.backend.sum(W ** 2, axis=1), axis=2)

        return loss_1 + loss_2 + loss_3 + contractive

    network.add_loss(loss(watermark, coverimage, decoder_output, marked, b1m, b1w, b2m, b2w))
    network.compile(optimizer='adam')

    print("network built successfully")
    return network


# Training model
def train_model(model, num_epochs, batch_size, path_to_save):
    watermark_dataset = utils.get_watermark_dataset(1800)
    print("watermark dataset get successfully")

    coverimage_dataset = utils.get_coverimage_dataset(1800, "C:/Users/aizee/OneDrive/Desktop/dataset/train")
    print("coverimage dataset get successfully")

    # Model training
    print("training...")
    history = model.fit([watermark_dataset, coverimage_dataset],
                        [watermark_dataset, coverimage_dataset],
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

    # Show graph of training
    graphs(history.history)


# Load model and display coverimage, watermark, marked, extracted_w
def predict_model(path_to_model, num_predicted):
    network = keras.models.load_model(path_to_model)

    watermark_dataset = utils.get_watermark_dataset(1800)
    coverimage_dataset = utils.get_coverimage_dataset(1800, "C:/Users/aizee/OneDrive/Desktop/dataset/train")

    for i in range(num_predicted):
        watermark = watermark_dataset[random.randint(0, 1800)]
        coverimage = coverimage_dataset[random.randint(0, 1800)]

        extracted, marked = network.predict(
            [watermark.reshape((1, 32, 32)), coverimage.reshape((1, 128, 128, 3))],
            batch_size=1
        )

        extracted = np.array(extracted).reshape((32, 32, 1))
        marked = np.array(marked).reshape((128, 128, 3))

        fig = plt.figure()
        sb = fig.add_subplot(2, 2, 1)
        sb.set_title("Исходное изображение")
        imshow(coverimage)

        sb = fig.add_subplot(2, 2, 2)
        sb.set_title("ЦВЗ")
        imshow(watermark)

        sb = fig.add_subplot(2, 2, 3)
        sb.set_title("Изображение с ЦВЗ")
        imshow(marked)

        sb = fig.add_subplot(2, 2, 4)
        sb.set_title("Извлеченный ЦВЗ")
        imshow(extracted, cmap='gray')

        print(str(i) + ":")
        print("\tМаксимальная разница м/у исходным и изобр. с ЦВЗ",
              np.max(np.abs(coverimage - marked)))
        print("\tМаксимальная разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
              np.max(np.abs(watermark - extracted)))
        print("\tСредняя разница м/у исходным и изобр. с ЦВЗ",
              np.mean(np.abs(coverimage - marked)))
        print("\tСредняя разница м/у исходным ЦВЗ и извлеченным ЦВЗ",
              np.mean(np.abs(watermark - extracted)))
    show()


# Set weights from full network to extractor submodel (invariance + extractor + decoder)
def create_extractor_model(path_to_full_model, path_to_save_extractor):
    model = keras.models.load_model(path_to_full_model)

    extractor = keras.Model(inputs=[model.outputs[1]], outputs=[model.outputs[0]], name="extractor")
    extractor.compile(optimizer="adam")

    for i in range(1, len(extractor.layers)):
        extractor.layers[i].set_weights(model.layers[i + 60].get_weights())

    extractor.save(path_to_save_extractor)
    print("extractor saved!")


# Show graphs loss(epoch) during training and validation
#   loss_dict - dictionary {'loss':[...], 'val_loss':[...]} (returned by keras model.predict method)
def graphs(loss_dict):
    fig = plt.figure()

    sb = fig.add_subplot(1, 1, 1)
    sb.set_title("Loss during training")
    sb.set_xlabel("Epoch")
    sb.set_ylabel("Loss value")
    sb.plot(np.arange(1, len(loss_dict['loss']) + 1, 1), loss_dict['loss'], color='blue', label='Training')
    sb.plot(np.arange(1, len(loss_dict['val_loss']) + 1, 1), loss_dict['val_loss'], color='green', label='Validation')
    sb.legend()

    show()


def load_model(path_to_model):
    return keras.models.load_model(path_to_model)
