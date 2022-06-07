import os
import model
import experiments
import tensorflow as tf


os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    # Create model and train
    # network = model.create_model()
    # model.train_model(network, 20, 1, 'full_network')

    # Show examples of trained model
    # model.predict_model('full_network', 3)

    # Create extractor model
    # model.create_extractor_model('full_network', 'extractor')

    # Experiments with noises
    # experiments.test_noise('gaussian', 'full_network', 'extractor')
    # experiments.test_noise('s&p', 'full_network', 'extractor')
    # experiments.test_noise('crop', 'full_network', 'extractor')
    return


main()
