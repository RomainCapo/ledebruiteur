"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import cv2
from tensorflow.keras.utils import Sequence
import multiprocessing as mp
import numpy as np
import os


class DataGenerator(Sequence):
    """Keras Sequence image data generator"""

    def __init__(self,
                 images_paths,
                 img_shape=(100, 100, 1),
                 batch_size=32,
                 shuffle=False):
        """Init

        Arguments:
            images_paths {Array} -- All image's path

        Keyword Arguments:
            img_shape {tuple} -- Image shape, channel last
            batch_size {int} -- Batch size (default: {32})
            shuffle {bool} -- Should shuffle the data (default: {False})
        """
        self.images_paths = images_paths
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Sequence length

        Returns:
            [int] -- Number of iterations per epoch
        """
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """Compute indexes on epoch end"""
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Get next

        Arguments:
            index {int} -- Index

        Returns:
            [(Array, Array)] -- Batch of images
        """
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        with mp.Pool() as pool:
            images = pool.starmap(
                cv2.imread, zip([self.images_paths.iloc[k, 0] for k in indexes], [cv2.IMREAD_GRAYSCALE] * self.batch_size))
            noised_images = pool.starmap(
                cv2.imread, zip([self.images_paths.iloc[k, 1] for k in indexes], [cv2.IMREAD_GRAYSCALE] * self.batch_size))


        images = np.array(images, np.float32) / 255
        noised_images = np.array(noised_images, np.float32) / 255

        y = images[..., np.newaxis]
        x = noised_images[..., np.newaxis]

        return x, y
