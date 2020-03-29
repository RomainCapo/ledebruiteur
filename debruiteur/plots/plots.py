"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from ..noise.noise import Noise


def plot_im_grid_from_df(df, noise=None, rows=5, columns=5, figsize=(8, 8)):
    """Plots a grid of ramdom sampled images

    Arguments:
        df {DataFrame} -- A dataframe with first column containing images's path

    Keyword Arguments:
        noise {Noise} -- Sublcass of Noise (default: {None})
        rows {int} -- Rows in image grid (default: {5})
        columns {int} -- Columns in image grid (default: {5})
        figsize {tuple} -- Subplot size (default: {(8, 8)})

    Raises:
        ValueError: Invalid noises
    """
    if not issubclass(type(noise), Noise) and noise is not None:
        raise ValueError("noise is not of valid type Noise")

    fig = plt.figure(figsize=figsize)
    sample = rows * columns

    for i, (idx, row) in enumerate(df.sample(sample).iterrows()):
        path = row[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if noise:
            img = noise.add(img)
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.axis('off')

    plt.show()


def plot_im_grid_from_list(images, rows, columns, figsize=(8, 8)):
    """Plots a grid of image from an image list

    Arguments:
        images {list} -- List of images
        rows {int} -- Number of rows in the grid
        columns {int} -- Number of columns in the grid

    Keyword Arguments:
        figsize {tuple} -- Subplot shape (default: {(8, 8)})
    """
    fig = plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.axis('off')

    plt.show()


def plot_model_loss(history):
    """Plots model's loss cuver

    Arguments:
        history {History} -- Keras training history callback
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_result_comparison(model, gen):
    """Plots comparison, between original, noised, denoised images

    Arguments:
        model {Model} -- Keras model
        gen {Sequence} -- Keras data generator
    """
    x, y = gen[0]
    y_pred = model.predict(x)

    images = np.vstack((y[:10], x[:10], y_pred[:10]))

    rows, cols = 10, 3

    fig = plt.figure(figsize=(10, 30))

    gs = gridspec.GridSpec(rows, cols, width_ratios=[
                           1]*cols, wspace=0.0, hspace=0.0)

    for i in range(rows):
        for j in range(cols):
            im = images[j * rows + i]
            im = im.reshape((im.shape[0], im.shape[1]))
            ax = plt.subplot(gs[i, j])
            ax.imshow(im, cmap=plt.cm.gray)
            ax.axis('off')

    plt.show()