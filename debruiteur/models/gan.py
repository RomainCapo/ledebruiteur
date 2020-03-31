"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""
from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, Dense, Input, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

from .blocks import convolutional_block, residual_block


class GAN():
    """Conditional GAN for image denoising"""

    def __init__(self, img_shape=(100, 100, 1)):
        """Initialize the discriminator and the generator

        Keyword Arguments:
            img_shape {tuple} -- Image shape (default: {(100, 100, 1)})
        """
        super().__init__()

        self.discriminator = self.build_discriminator(img_shape)
        self.discriminator_opt = Adam()

        self.generator = self.build_generator(img_shape)
        self.generator_opt = Adam()

        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.mean_squared_error = MeanSquaredError()

    def build_generator(self, input_shape=(100, 100, 1)):
        """Builds the generator

        Keyword Arguments:
            input_shape {tuple} -- Image shape (default: {(100, 100, 1)})

        Returns:
            Model -- The generator model
        """
        X_input = Input(input_shape)

        X = convolutional_block(
            X_input, filter=32, k_size=9, act_layer_name="gen_conv1_relu")
        X_shortcut = X
        X = convolutional_block(X, filter=64, k_size=3,
                                act_layer_name="gen_conv2_relu")

        X = residual_block(X, filter=64, k_size=3, act_layer_names=(
            "gen_res1_conv1_relu", "gen_res1_conv2_relu"))

        X = Conv2DTranspose(filters=32, kernel_size=(
            3, 3), strides=(1, 1), padding="same")(X)

        X = Add()([X, X_shortcut])

        X = convolutional_block(
            X, filter=1, k_size=9, act_layer_name="gen_conv4_tanh", activation="tanh")

        X = Add()([X, X_input])
        X = Lambda(lambda x: (x + 1) / 2)(X)  # as tanh in ]-1; 1[

        model = Model(inputs=X_input, outputs=X)

        return model

    def build_discriminator(self, input_shape=(100, 100, 1)):
        """Builds the discriminator

        Keyword Arguments:
            input_shape {tuple} -- Image shape (default: {(100, 100, 1)})

        Returns:
            Model -- The discriminator model
        """
        X_input = Input(input_shape)

        X = convolutional_block(X_input, filter=48, k_size=4,
                                act_layer_name="disc_conv1_relu", stride=2)
        X = convolutional_block(X, filter=96, k_size=4,
                                act_layer_name="disco_conv2_relu", stride=2)
        X = Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1))(X)
        X = Flatten()(X)
        X = Dense(units=1, activation="sigmoid")(X)

        model = Model(inputs=X_input, outputs=X)

        return model

    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss, cross entropy

        Arguments:
            real_output {Array} -- Discriminator predictions on real images
            fake_output {Array} -- Discriminator predictions on fake images

        Returns:
            [type] -- [description]
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, real_images, gen_images, fake_output):
        """Generator loss, uses mean squared error and binary cross entropy

        Arguments:
            real_image {Array} -- Real images
            gen_images {Array} -- Fake images
            fake_output {Array} -- Discriminator prediction on fake images

        Returns:
            float -- Loss
        """
        return self.mean_squared_error(real_images, gen_images) + self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def generate_and_plot_images(self, epoch, test_input):
        """Generate fake image and plot them

        Arguments:
            epoch {int} -- Current epoch
            test_input {Array} -- Noised images
        """
        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(8, 8))

        for i in range(predictions.shape[0]):
            fig.add_subplot(4, 2, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.show()

    @tf.function()
    def train_step(self, noised_images, images):
        """Perform a train step for a batch

        Arguments:
            noised_images {Array} -- Images with nois
            images {Array} -- Original images
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noised_images, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(
                images, generated_images, fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables)

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_opt.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_opt.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, train_gen, val_gen, epochs=10):
        """Trains the gan

        Arguments:
            train_gen {Sequence} -- Train data generator
            val_gen {Sequence} -- Validation data generator

        Keyword Arguments:
            epochs {int} -- Epochs (default: {10})

        Returns:
            tuple -- (train loss history, validation loss history)
        """

        for epoch in range(1, epochs + 1):

            print(f"Epoch {epoch}/{epochs}")

            num_batches = len(train_gen)
            progress_bar = Progbar(target=num_batches)

            for index, (X_train, y_train) in enumerate(train_gen):
                self.train_step(X_train, y_train)
                progress_bar.update(index + 1)

            display.clear_output(wait=True)
            self.generate_and_plot_images(epoch, val_gen[epoch][0])
