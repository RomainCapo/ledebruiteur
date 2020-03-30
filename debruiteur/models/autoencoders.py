"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, UpSampling2D
from tensorflow.keras.models import Sequential


def build_conv_autoencoder(input_shape=(100, 100, 1), optimizer="adam", loss="mse"):
    """Autoencode model, optimizer : adam, loss : MSE

    Keyword Arguments:
        input_shape {tuple} -- Image shape, grayscale (default: {(100, 100, 1)})
        optimizer {optimizer} -- valid keras optimizer (default: {"adam"})
        loss {loss} -- valid keras loss (default: {"mse"})

    Returns:
        Model -- Keras convolutional autoencoder model
    """
    ae_model = Sequential()

    # Encoder
    ae_model.add(Conv2D(64, (3, 3), input_shape=input_shape,
                        activation="relu", padding="same"))
    ae_model.add(MaxPool2D((2, 2), padding="same"))
    ae_model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    ae_model.add(MaxPool2D((2, 2), padding="same"))
    ae_model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
    ae_model.add(MaxPool2D((2, 2), padding="same"))
    ae_model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))

    # Decoder
    ae_model.add(UpSampling2D((2, 2)))
    ae_model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    ae_model.add(UpSampling2D((2, 2)))
    ae_model.add(Conv2D(64, (3, 3), activation="relu"))
    ae_model.add(UpSampling2D((2, 2)))
    ae_model.add(Conv2D(1, (3, 3), padding="same", activation="sigmoid"))

    ae_model.compile(optimizer=optimizer, loss=loss)

    return ae_model


def build_dense_autoencoder(input_shape=10000, optimizer="adam", loss="binary_crossentropy"):
    """Autoencode model, optimizer : adam, loss : MSE

    Keyword Arguments:
        input_shape {tuple} -- Image shape, grayscale (default: {1000})
        optimizer {optimizer} -- valid keras optimizer (default: {"adam"})
        loss {loss} -- valid keras loss (default: {"mse"})

    Returns:
        Model -- Keras dense autoencoder model
    """
    ae_model = Sequential()

    # Encoder
    ae_model.add(Dense(units=512, activation='relu', input_shape=(input_shape, )))
    ae_model.add(Dense(units=256, activation='relu'))
    ae_model.add(Dense(units=128, activation='relu'))

    # Decoder
    ae_model.add(Dense(units=256, activation='relu'))
    ae_model.add(Dense(units=512, activation='relu'))
    ae_model.add(Dense(units=input_shape, activation='sigmoid'))

    ae_model.compile(optimizer=optimizer, loss=loss)

    return ae_model
