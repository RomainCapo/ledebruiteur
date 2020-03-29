"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

from keras.initializers import RandomNormal
from keras.layers import Activation, BatchNormalization, Conv2D
from keras.layers.merge import Add


def convolutional_block(X, filter, k_size, stride=1, activation="relu"):
    """Convolutional block, Conv2D -> BatchNormalization -> Activation

    Arguments:
        X {Tensor} -- Input tensor
        filter {int} -- Number of filters
        k_size {int} -- Kernel size
        stride {int} -- Stride size

    Returns:
        Tensor -- Output tensor after convolutional block
    """

    X = Conv2D(filters=filter, kernel_size=(k_size, k_size), strides=(stride, stride),
               padding="same", kernel_initializer=RandomNormal(stddev=0.03, seed=None))(X)
    X = BatchNormalization(axis=3)(X)
    x = Activation(activation)(X)

    return X


def residual_block(X, filter, k_size, stride):
    """Residual block, convolutional_block -> convolutional_block > out + x

    Arguments:
        X {Tensor} -- Input tensor
        filter {int} -- Number of filters
        k_size {int} -- Kernel size
        stride {int} -- Stride size

    Returns:
        Tensor -- Output tensor after residual block
    """

    X_shortcut = X

    X = convolutional_block(X, filter, k_size, stride)
    X = convolutional_block(X, filter, k_size, stride)

    X = Add()([X, X_shortcut])

    return X
