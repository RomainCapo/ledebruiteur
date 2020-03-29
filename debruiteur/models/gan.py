from keras.initializers import glorot_uniform
from keras.layers import Activation, Conv2D, Conv2DTranspose, Input, Lambda
from keras.layers.merge import Add
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

from .blocks import convolutional_block, residual_block
from .loss import generator_loss


class GAN():

    def __init__(self, img_shape=(100, 100, 1)):
        super().__init__()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            optimizer="Adam", loss="binary_crossentropy")

        self.generator = self.build_generator()

        """
        z = Input(shape=(None, *img_shape))
        fake_img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(fake_img)

        optimizer = Adam(0.0002, 0.5)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)"""

    def build_generator(self, input_shape=(100, 100, 1)):
        X_input = Input(input_shape)

        X = convolutional_block(X_input, filter=32, k_size=9)
        X_shortcut = X
        X = convolutional_block(X, filter=64, k_size=3)
        X = convolutional_block(X, filter=128, k_size=3)

        X = convolutional_block(X, filter=128, k_size=3)
        X = convolutional_block(X, filter=128, k_size=3)
        X = convolutional_block(X, filter=128, k_size=3)

        X = Conv2DTranspose(filters=64, kernel_size=(
            3, 3), strides=(1, 1), padding="same")(X)
        X = Conv2DTranspose(filters=32, kernel_size=(
            3, 3), strides=(1, 1), padding="same")(X)

        X = Add()([X, X_shortcut])

        X = convolutional_block(X, filter=1, k_size=9, activation="tanh")

        X = Add()([X, X_input])
        X = Lambda(lambda x: (x + 1) / 2)(X)  # as tanh in ]-1; 1[

        model = Model(inputs=X_input, outputs=X)

        return model

    def build_discriminator(self, input_shape=(100, 100, 1)):
        X_input = Input(input_shape)

        X = convolutional_block(X_input, filter=48, k_size=4, stride=2)
        X = convolutional_block(X, filter=96, k_size=4, stride=2)
        X = convolutional_block(X, filter=192, k_size=4, stride=2)
        X = convolutional_block(X, filter=384, k_size=4, stride=1)
        X = Conv2D(filters=1, kernel_size=(13, 13), strides=(1, 1))(X)
        X = Activation("sigmoid")(X)

        model = Model(inputs=X_input, outputs=X)

        return model

    def train(self, train_gen, val_gen, batch_size=32, epochs=10):
        # TODO : Nice loss log
        valid = np.ones((batch_size, 1))  # When true image class is 0
        fake = np.zeros((batch_size, 1))  # When fake image class is 1

        for epoch in range(epochs):

            # X_train : noised images, y_train : original images
            for X_train, y_train in train_gen:
                Gz = self.generator.predict(X_train)

                # retourne la perte mais il nous faut la prédiction
                Dx = self.discriminator.train_on_batch(X_train, valid)  # Classify original images
                Dg = self.discriminator.train_on_batch(Gz, fake) # Classify fake images

                Dg = self.discriminator.predict(X_train)

                gen_loss = generator_loss(Gz, y_train, Dg)
                self.generator.compile(optimizer="Adam", loss=gen_loss)

                self.generator.train_on_batch(X_train, y_train)

            for X_val, y_val in val_gen:
                Gz = self.generator.predict(X_train)

                Dg = self.discriminator.predict(X_val)

                gen_loss = generator_loss(Gz, y_train, Dg)
                self.generator.compile(optimizer="Adam", loss=gen_loss)
                g_loss = self.generator.test_on_batch(X_train, y_train)