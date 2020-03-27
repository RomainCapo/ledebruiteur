import multiprocessing as mp
from keras.utils import Sequence
import numpy as np
import os
import cv2

class DataGenerator(Sequence):

    def __init__(self,
                 images_paths,
                 seq=None,
                 batch_size=32,
                 image_dimensions=(100, 100, 3),
                 shuffle=False):
        self.images_paths = images_paths
        self.seq = seq
        self.dim = image_dimensions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        with mp.Pool() as pool:
            images = pool.map(
                cv2.imread, [self.images_paths[k] for k in indexes])

        images = np.array(images, np.float32) / 255

        return images
