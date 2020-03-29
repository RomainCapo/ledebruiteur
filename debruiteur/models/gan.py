from keras.models import Model, Sequential
from .blocks import identity_block

class GAN():

    def __init__(self):
        super().__init__()

    def build_generator(self):
        generator = Sequential() # TODO utiliser la version Functional
        # TODO autoencoder with residual block
        generator.summary()
        pass

    def build_discriminator(self):
        discriminator = Sequential()
        # TODO Convolution for binary classification
        pass

    def train(self):
        # TODO custom loss from paper
        pass
