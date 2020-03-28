"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import os
import shutil
import cv2
import matplotlib.pyplot as plt
from ..noise.noise import Noise


def init_dir(directory):
    """Creates a directory if not exists and remove content if it exist

    Arguments:
        directory {string} -- path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)


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
        img = cv2.imread(path)
        if noise:
            img = noise.add(img)
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
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
        plt.imshow(img)
        plt.axis('off')

    plt.show()
