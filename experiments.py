import copy
import numpy as np
from tensorflow import keras
import utils
from matplotlib import pyplot as plt
from skimage.io import imshow, show
from skimage.util import random_noise


# Get image with Gaussian blur
def add_gaussian_noise(image):
    return random_noise(image, 'gaussian', var=0.001)


# Get image with a few pixels replaced
def add_salt_and_pepper_noise(image, intensity):
    noised_image = copy.deepcopy(image)

    noise = random_noise(np.full((128, 128, 3), -1), 's&p', amount=intensity)

    for i in range(0, 128):
        for j in range(0, 128):
            for k in range(0, 3):
                if noise[i][j][k] == 1:
                    noised_image[i][j] = [1., 1., 1.]
                elif noise[i][j][k] == -1:
                    noised_image[i][j] = [0., 0., 0.]

    return noised_image


# Get image with two rectangles cut out
def add_cropping_noise(image):
    noised_image = copy.deepcopy(image)

    for k in range(0, 2):
        left_up_corner_i = np.random.randint(0, 90)
        left_up_corner_j = np.random.randint(0, 60)

        width = np.random.randint(30, 60)
        height = np.random.randint(30, 60)

        for i in range(left_up_corner_i, left_up_corner_i + height):
            for j in range(left_up_corner_j, left_up_corner_j + width):
                if i >= 127 or j >= 127:
                    break
                noised_image[i][j] = [0., 0., 0.]

    return noised_image


# Applying noise to marked image and trying to extract watermark
#   noise - type of noise ('gaussian', 's&p', 'crop')
def test_noise(noise, path_to_full, path_to_extractor):
    network = keras.models.load_model(path_to_full)
    extractor = keras.models.load_model(path_to_extractor)

    watermark = utils.get_watermark_dataset(50)[np.random.randint(0, 50)]
    coverimage = utils.get_coverimage_dataset(
        50,
        "C:/Users/aizee/OneDrive/Desktop/dataset/train"
    )[np.random.randint(0, 50)]

    extracted, marked = network.predict(
        [watermark.reshape((1, 32, 32, 1)), coverimage.reshape((1, 128, 128, 3))],
        batch_size=1
    )

    extracted = np.array(extracted).reshape((32, 32, 1))
    marked = np.array(marked).reshape((128, 128, 3))

    noised_marked = None
    if noise == 'gaussian':
        noised_marked = add_gaussian_noise(marked)
    elif noise == 's&p':
        noised_marked = add_salt_and_pepper_noise(marked, 0.005)
    elif noise == 'crop':
        noised_marked = add_cropping_noise(marked)

    noised_extracted = extractor.predict(noised_marked.reshape((1, 128, 128, 3)), batch_size=1)
    noised_extracted = np.array(noised_extracted).reshape((32, 32, 1))

    fig = plt.figure()
    sb = fig.add_subplot(2, 2, 1)
    sb.set_title("Изображение с ЦВЗ")
    imshow(marked)

    sb = fig.add_subplot(2, 2, 2)
    sb.set_title("Извлеченный ЦВЗ")
    imshow(extracted, cmap='gray')

    sb = fig.add_subplot(2, 2, 3)
    sb.set_title("Зашумленное изображение с ЦВЗ")
    imshow(noised_marked.reshape((128, 128, 3)))

    sb = fig.add_subplot(2, 2, 4)
    sb.set_title("Извлеченный ЦВЗ из зашумленного")
    imshow(noised_extracted.reshape((32, 32, 1)), cmap='gray')

    print("Максимальная разница м/у ЦВЗ и зашумленным ЦВЗ: ", np.max(np.abs(extracted - noised_extracted)))
    print("Средняя разница м/у ЦВЗ и зашумленным ЦВЗ: ", np.mean(np.abs(extracted - noised_extracted)))

    show()
