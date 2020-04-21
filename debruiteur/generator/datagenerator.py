"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import os

import cv2
import gc
import numpy as np
from tensorflow.keras.utils import Sequence

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
        gc.collect()

    def __getitem__(self, index):
        """Get next

        Arguments:
            index {int} -- Index

        Returns:
            [(Array, Array)] -- Batch of images
        """
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        images = [cv2.imread(self.images_paths.iloc[k, 0],
                             cv2.IMREAD_GRAYSCALE) for k in indexes]
        noised_images = [cv2.imread(
            self.images_paths.iloc[k, 1], cv2.IMREAD_GRAYSCALE) for k in indexes]

        images = np.array(images, np.float32) / 255
        noised_images = np.array(noised_images, np.float32) / 255

        if len(self.img_shape) == 3:
            images = images[..., np.newaxis]
            noised_images = noised_images[..., np.newaxis]
        else:
            images = images.reshape(-1, *self.img_shape)
            noised_images = noised_images.reshape(-1, *self.img_shape)

        return noised_images, images