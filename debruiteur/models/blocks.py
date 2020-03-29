from keras.initializers import glorot_uniform
from keras.layers import Activation, BatchNormalization, Conv2D
from keras.layers.merge import Add

def convolutional_block(X, f, filters):
    """Convolutional block from ResNet architecture

    Arguments:
        X {Tensor} -- Input tensor
        f {tuple} -- Tuple containing number of filter for each three convolution layers
        filters {int} -- kernel dimension for middle convolution layer

    Raises:
        ValueError: Invalid tuple of number of filters given

    Returns:
        Tensor -- Output tensor after convolutional block
    """
    if type(filters) != list or len(filters) != 3:
        raise ValueError("There should be 3 given filters quantity")

    X_shortcut = X

    f1, f2, f3 = filters

    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    x = Activation("relu")(X)

    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1),
               padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    x = Activation("relu")(X)

    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def indentity_block(X, f, filters, s=2):
    """Identity block from ResNet architecture

    Arguments:
        X {Tensor} -- Input tensor
        f {tuple} -- Tuple containing number of filter for each three convolution layers
        filters {int} -- kernel dimension for middle convolution layer

    Keyword Arguments:
        s {int} -- Strides dimension of middle convolution layer (default: {2})

    Raises:
        ValueError: Invalid tuple of number of filters given

    Returns:
        Tensor -- Output tensor after identity block
    """
    if type(filters) != list or len(filters) != 3:
        raise ValueError("There should be 3 given filters quantity")

    X_shortcut = X

    f1, f2, f3 = filters

    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    x = Activation("relu")(X)

    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1),
               padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    x = Activation("relu")(X)

    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
