"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D
from tensorflow.keras.layers.merge import Add


def convolutional_block(X, filter, k_size, act_layer_name, stride=1, activation="relu"):
    """Convolutional block, Conv2D -> BatchNormalization -> Activation

    Arguments:
        X {Tensor} -- Input tensor
        filter {int} -- Number of filters
        k_size {int} -- Kernel size
        act_layer_names {String} -- Activation layer name

    Keyword Arguments:
        stride {int} -- Stride size

    Returns:
        Tensor -- Output tensor after convolutional block
    """
    X = Conv2D(filters=filter, kernel_size=(k_size, k_size), strides=(stride, stride),
               padding="same", kernel_initializer=RandomNormal(stddev=0.03, seed=None))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation, name=act_layer_name)(X)

    return X


def residual_block(X, filter, k_size, act_layer_names):
    """Residual block, convolutional_block -> convolutional_block > out + x

    Arguments:
        X {Tensor} -- Input tensor
        filter {int} -- Number of filters
        k_size {int} -- Kernel size
        act_layer_names {tuple} -- Tuple of strings

    Raises:
            ValueError: Two names for the block activation layers must be provided

    Returns:
        Tensor -- Output tensor after residual block
    """
    if len(act_layer_names) != 2:
        raise ValueError(
            "Two names for the block activation layers must be provided")

    X_shortcut = X

    X = convolutional_block(X, filter, k_size, act_layer_names[0])
    X = convolutional_block(X, filter, k_size, act_layer_names[1])

    X = Add()([X, X_shortcut])

    return X
