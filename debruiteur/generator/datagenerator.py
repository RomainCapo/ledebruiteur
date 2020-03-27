import multiprocessing as mp
from keras.utils import Sequence
import numpy as np
import os
import cv2


class DataGenerator(Sequence):
    """Keras Sequence image data generator"""

    def __init__(self,
                 images_paths,
                 batch_size=32,
                 image_dimensions=(100, 100, 3),
                 shuffle=False):
        """Init

        Arguments:
            images_paths {Array} -- All image's path

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {32})
            image_dimensions {tuple} -- Image shape (default: {(100, 100, 3)})
            shuffle {bool} -- Should shuffle the data (default: {False})
        """
        self.images_paths = images_paths
        self.dim = image_dimensions
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
            index {[type]} -- Index

        Returns:
            [Array] -- Batch of images
        """
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        with mp.Pool() as pool:
            images = pool.map(
                cv2.imread, [self.images_paths[k] for k in indexes])

        images = np.array(images, np.float32) / 255

        return images
