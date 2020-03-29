import keras.backend as K


def generator_loss(y_true, y_pred, Dg):
    """Generator loss

    Arguments:
        y_true {Array} -- Ground truth
        y_pred {Array} -- Predicted array

    Returns:
        float -- Generator loss
    """

    def loss(y_true, y_pred):
        return 0.5 * adversial_loss(Dg) + pixel_loss(y_true, y_pred) + style_loss(y_true, y_pred) # + smooth_loss(Gz)

    return loss


def pixel_loss(y_true, y_pred):
    """Pixel loss l2 loss of pixelwise differences

    Arguments:
        y_true {Array} -- Ground truth
        y_pred {Array} -- Predicted array

    Returns:
        float -- RMSE
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def style_loss(y_true, y_pred):
    """Style loss

    Arguments:
        y_true {Array} -- Ground truth
        y_pred {Array} -- Predicted array]

    Returns:
        float -- style loss
    """
    S = gram_matrix(y_true)
    C = gram_matrix(y_pred)
    channels = 1
    size = 100 * 100
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def gram_matrix(x):
    """Gram matrix

    Arguments:
        x {Array} -- Input array

    Returns:
        array -- Gram matrix
    """
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def adversial_loss(x):
    """Adversial loss

    Arguments:
        x {Array} -- Input array

    Returns:
        float -- Adversial loss
    """
    return -K.mean(K.log(x))


def smooth_loss(x):
    b, w, h, c = x.shape
    # TODO
    pass
