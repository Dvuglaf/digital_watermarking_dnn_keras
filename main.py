import os
from tensorflow import keras
import tensorflow as tf
import tensorflow
import numpy as np

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


def get_model():
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
    encoder_output = tensorflow.reshape(x, [-1, 128, 128, 3])

    # Embedder
    x, b1w, b2w = conv_block_b_star(encoder_output)
    x = keras.layers.Concatenate(axis=len(x.shape) - 1)([x, coverimage])
    x = conv_block_b(x)
    marked = keras.layers.Conv2D(3, (1, 1), padding='same', name='2', activation='sigmoid', input_shape=x.shape[1:])(x)
    embedder_output_1 = keras.layers.Conv2D(3, (1, 1), padding='same', activation='relu', input_shape=x.shape[1:])(x)
    embedder_output_2 = embedder_output_1
#history dict: {'loss': [0.560004711151123, 0.41972488164901733, 0.4167323112487793, 0.41443121433258057, 0.41267696022987366, 0.41153934597969055, 0.41080740094184875, 0.41018927097320557, 0.40962815284729004, 0.4091819226741791, 0.4088590145111084, 0.4011174738407135, 0.28584182262420654, 0.28520411252975464, 0.2850855588912964, 0.28514400124549866, 0.2846873104572296, 0.28472182154655457, 0.2845335900783539, 0.2842155694961548], 'val_loss': [0.41678234934806824, 0.4131079614162445, 0.4219195544719696, 0.4116959273815155, 0.40799906849861145, 0.410060852766037, 0.40874943137168884, 0.40750887989997864, 0.4098321795463562, 0.4051806926727295, 0.40662071108818054, 0.2917393743991852, 0.2830906808376312, 0.28553032875061035, 0.2819216251373291, 0.2823534607887268, 0.2833417057991028, 0.28384754061698914, 0.28309953212738037, 0.2815670371055603]}


    # Invariance network
    N = 5
    # embedder_output_1 = keras.layers.BatchNormalization()(embedder_output_1)
    x = keras.layers.Dropout(0.25)(embedder_output_1)
    invariance_output = keras.layers.Dense(N, activation='relu', name='3')(x)

    # Extractor
    x = conv_block_b(invariance_output)
    x = conv_block_b(x)
    extractor_output = keras.layers.Conv2D(3, (1, 1), padding='same', name='4', activation='relu',
                                           input_shape=x.shape[1:])(x)

    # Decoder
    x = tensorflow.reshape(extractor_output, [-1, 32, 32, 48])
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
                keras.backend.mean(keras.backend.abs(gram_matrix(b_1_w) - gram_matrix(b_1_m))) +
                keras.backend.mean(keras.backend.abs(gram_matrix(b_2_w) - gram_matrix(b_2_m)))
        )
        W = keras.backend.variable(value=network.get_layer('3').get_weights()[0])  # N x N_hidden
        W = keras.backend.transpose(W)  # N_hidden x N
        h = network.get_layer('3').output
        dh = 1 - h * h  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = 0.01 * keras.backend.sum(dh ** 2 * keras.backend.sum(W ** 2, axis=1), axis=2)

        return loss_1 + loss_2 + loss_3 + contractive

    network.add_loss(loss(watermark, coverimage, decoder_output, marked, b1m, b1w, b2m, b2w))
    return network


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
    network = get_model()
    network.compile(optimizer='adam')
    print("network built successfully")

    watermark_dataset = get_watermark_dataset(1800)
    print("watermark dataset get successfully")
    coverimage_dataset = get_coverimage_dataset(1800)
    print("coverimage dataset get successfully")
    print("training...")
    # Model training
    history = network.fit([watermark_dataset, coverimage_dataset], [watermark_dataset, coverimage_dataset],
                          epochs=20,
                          batch_size=2,
                          validation_split=0.2,
                          shuffle=True
                          )
    print("training ends")
    print('history dict:', history.history)

    # Save model
    network.save('full')
    print("FULL SAVED!")

    # Set weights from full network to submodels (encoder + embedder & invariance + extractor + decoder)
    embedder = keras.Model(inputs=[network.inputs[0], network.inputs[1]], outputs=[network.outputs[1]], name="embedder")
    embedder.compile(optimizer="adam")
    extractor = keras.Model(inputs=[network.outputs[1]], outputs=[network.outputs[0]], name="extractor")
    extractor.compile(optimizer="adam")

    for i in range(0, len(embedder.layers), 1):
        embedder.layers[i].set_weights(network.layers[i].get_weights())
    for i in range(len(embedder.layers) + 1, len(embedder.layers) + len(extractor.layers)):
        extractor.layers[i - len(embedder.layers)].set_weights(network.layers[i - 1].get_weights())

    embedder.save("embedder")
    extractor.save("extractor")
    print("embedder SAVED!")
    print("extractor SAVED!")


main()


# print('\n# Оцениваем на тестовых данных')
# results = network.evaluate([x_test, c_test], [x_test, c_test], batch_size=3)
# print('test loss, test acc:', results)

#history dict: {'loss': [0.8572868704795837, 0.7147248387336731, 0.7113635540008545, 0.7092450857162476, 0.7076516151428223, 0.706613302230835, 0.706193208694458, 0.7058060169219971, 0.5288805365562439, 0.4546271860599518, 0.45479029417037964, 0.45418810844421387, 0.454147070646286, 0.45387157797813416, 0.45381855964660645, 0.4534739851951599, 0.453430712223053, 0.4532884657382965, 0.45318394899368286, 0.453174889087677, 0.45284149050712585, 0.4528637230396271, 0.4526495039463043, 0.4526776671409607, 0.4526442885398865, 0.45252525806427, 0.4523104727268219, 0.4522817134857178, 0.4522295892238617, 0.45240625739097595, 0.452180951833725, 0.4521979093551636, 0.4520820379257202, 0.45221856236457825, 0.4519989490509033, 0.4519577622413635, 0.4519959092140198, 0.4519016444683075, 0.4518061578273773, 0.45175400376319885, 0.45185956358909607, 0.4517829120159149, 0.45155981183052063, 0.45165589451789856, 0.4517591893672943, 0.4516826570034027, 0.45165157318115234, 0.45167335867881775, 0.45151466131210327, 0.4515877366065979], 'val_loss': [0.7208779454231262, 0.708084225654602, 0.7080684900283813, 0.7053614258766174, 0.7103806138038635, 0.7074121832847595, 0.704506516456604, 0.7018365859985352, 0.45207929611206055, 0.4527471661567688, 0.45119524002075195, 0.45090240240097046, 0.45212018489837646, 0.45266517996788025, 0.45284342765808105, 0.45099329948425293, 0.45161178708076477, 0.45112740993499756, 0.450734406709671, 0.45014193654060364, 0.45099732279777527, 0.45103225111961365, 0.4493444561958313, 0.45098230242729187, 0.4499287009239197, 0.45061999559402466, 0.44959211349487305, 0.45144280791282654, 0.4500782787799835, 0.44952327013015747, 0.4499365985393524, 0.44929367303848267, 0.45022737979888916, 0.45055484771728516, 0.4501899778842926, 0.4505578577518463, 0.44970712065696716, 0.4499427378177643, 0.44922536611557007, 0.4492485225200653, 0.4494638442993164, 0.4495351314544678, 0.44995754957199097, 0.44930896162986755, 0.44891634583473206, 0.4488832652568817, 0.4490947127342224, 0.4490487277507782, 0.4494324028491974, 0.44934096932411194]}
#history dict: {'loss': [1.3439278602600098, 0.3348102271556854, 0.3181597590446472, 0.3088144063949585, 0.3026980459690094, 0.3006627857685089, 0.29890376329421997, 0.2974616587162018, 0.29581210017204285, 0.2944021224975586, 0.2932160794734955, 0.2923561632633209, 0.29184162616729736, 0.2910248339176178, 0.29032522439956665, 0.2896820604801178, 0.28895601630210876, 0.28861600160598755, 0.2881071865558624, 0.28746169805526733], 'val_loss': [0.3960309624671936, 0.32817354798316956, 0.31868812441825867, 0.30187422037124634, 0.2990926206111908, 0.29631808400154114, 0.2937230169773102, 0.2933374047279358, 0.29191482067108154, 0.29022708535194397, 0.2908184230327606, 0.28882619738578796, 0.28801625967025757, 0.28768283128738403, 0.2888622283935547, 0.2898631691932678, 0.28668585419654846, 0.2870660722255707, 0.285495787858963, 0.2858416736125946]}

#TODO: Модуль Utils

