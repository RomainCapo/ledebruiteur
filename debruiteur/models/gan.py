"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""
from collections import defaultdict
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, Dense, Input, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
import numpy as np

from .blocks import convolutional_block, residual_block
from .loss import generator_loss


class GAN():
    """Conditional GAN for image denoising"""

    def __init__(self, img_shape=(100, 100, 1)):
        """Initialize the discriminator and the generator

        Keyword Arguments:
            img_shape {tuple} -- Image shape (default: {(100, 100, 1)})
        """
        super().__init__()

        self.discriminator = self.build_discriminator(img_shape)
        self.discriminator.compile(
            optimizer="Adam", loss="binary_crossentropy")

        self.generator = self.build_generator(img_shape)
        self.generator.compile(optimizer="Adam", loss="mean_squared_error")

        """self.generator_outputs_dict = dict(
            [(layer.name, layer.output) for layer in self.generator.layers])"""

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
        """X = convolutional_block(X, filter=128, k_size=3,
                                act_layer_name="gen_conv3_relu")"""

        X = residual_block(X, filter=64, k_size=3, act_layer_names=(
            "gen_res1_conv1_relu", "gen_res1_conv2_relu"))
        """X = residual_block(X, filter=128, k_size=3, act_layer_names=(
            "gen_res2_conv1_relu", "gen_res2_conv2_relu"))
        X = residual_block(X, filter=128, k_size=3, act_layer_names=(
            "gen_res3_conv1_relu", "gen_res3_conv2_relu"))"""

        X = Conv2DTranspose(filters=32, kernel_size=(
            3, 3), strides=(1, 1), padding="same")(X)
        """X = Conv2DTranspose(filters=16, kernel_size=(
            3, 3), strides=(1, 1), padding="same")(X)"""

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
        """X = convolutional_block(X, filter=192, k_size=4,
                                act_layer_name="disco_conv3_relu", stride=2)
        X = convolutional_block(X, filter=384, k_size=4,
                                act_layer_name="disco_conv4_relu", stride=1)"""
        X = Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1))(X)
        X = Flatten()(X)
        X = Dense(units=1, activation="sigmoid")(X)

        model = Model(inputs=X_input, outputs=X)

        return model

    def get_feature_layers(self):
        """Gets the feature layers

        Returns:
            tuple -- Tuple of array coinaining (style features, combination features)
        """
        feature_layers = ["gen_conv1_relu", "gen_conv2_relu",
                          "gen_res1_conv1_relu", "gen_res1_conv2_relu"]

        comb_features = []
        style_features = []
        for layer_name in feature_layers:
            layer_features = self.generator_outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            style_features.append(style_reference_features)
            combination_features = layer_features[2, :, :, :]
            comb_features.append(combination_features)

        return style_features, comb_features

    def train(self, train_gen, val_gen, batch_size=32, epochs=10):
        """Trains the gan

        Arguments:
            train_gen {Sequence} -- Train data generator
            val_gen {Sequence} -- Validation data generator

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {32})
            epochs {int} -- Epochs (default: {10})

        Returns:
            tuple -- (train loss history, validation loss history)
        """
        valid = np.ones((batch_size, 1))  # When true image class is 0
        fake = np.zeros((batch_size, 1))  # When fake image class is 1

        train_history = defaultdict(list)
        val_history = defaultdict(list)

        for epoch in range(1, epochs + 1):

            print(f"Epoch {epoch}/{epochs}")

            num_batches = len(train_gen)
            progress_bar = Progbar(target=num_batches)

            epoch_train_gen_loss = []
            epoch_train_disc_loss = []
            epoch_val_gen_loss = []
            epoch_val_disc_loss = []

           # X_train : noised images, y_train : original images
            for index, (X_train, y_train) in enumerate(train_gen):
                Gz = self.generator.predict(X_train)

                Dx = self.discriminator.train_on_batch(
                    X_train, valid)  # Classify original images
                Dg = self.discriminator.train_on_batch(
                    Gz, fake)  # Classify fake images
                epoch_train_disc_loss.append(Dg)

                """
                style_features, comb_features = self.get_feature_layers()
                gen_loss = generator_loss(
                    y_train, Gz, Dg, style_features, comb_features)
                self.generator.compile(
                    optimizer="Adam", loss="mse", experimental_run_tf_function=False)"""

                g_loss = self.generator.train_on_batch(X_train, y_train)
                epoch_train_gen_loss.append(g_loss)

                progress_bar.update(index + 1)

            discriminator_train_loss = np.mean(
                np.array(epoch_train_disc_loss), axis=0)
            generator_train_loss = np.mean(
                np.array(epoch_train_gen_loss), axis=0)

            train_history["generator"].append(generator_train_loss)
            train_history["discriminator"].append(discriminator_train_loss)

            print(f"Validation for epoch {epoch}")

            for X_val, y_val in val_gen:
                Gz = self.generator.predict(X_val)

                Dg = self.discriminator.evaluate(Gz, fake)
                epoch_val_disc_loss.append(Dg)
                """
                style_features, comb_features = self.get_feature_layers()
                gen_loss = generator_loss(
                    y_val, Gz, Dg, style_features, comb_features)
                self.generator.compile(optimizer="Adam", loss=gen_loss)"""

                g_loss = self.generator.test_on_batch(X_val, y_val)
                epoch_val_gen_loss.append(g_loss)

            discriminator_val_loss = np.mean(
                np.array(epoch_val_disc_loss), axis=0)
            generator_val_loss = np.mean(np.array(epoch_val_gen_loss), axis=0)

            val_history["generator"].append(generator_val_loss)
            val_history["discriminator"].append(discriminator_val_loss)

            print(f"Train generator loss {train_history['generator'][-1]}")
            print(
                f"Train discriminator loss {train_history['discriminator'][-1]}")
            print(f"Validation generator loss {val_history['generator'][-1]}")
            print(
                f"Validation generator loss {val_history['discriminator'][-1]}")

            gc.collect()

        return train_history, val_history
