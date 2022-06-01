import utils
import model


def main():
    utils.init()

    #network = model.create_model()
    #model.train_model(network, 15, 1, 'full_network')

    #utils.create_submodels(network)

    model.predict_model('full', 3)


main()
