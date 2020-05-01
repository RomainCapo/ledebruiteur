"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import random

import numpy as np
import cv2
from abc import ABC, abstractmethod


class Noise(ABC):

    @abstractmethod
    def add(self, img):
        """Add noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Raises:
            NotImplementedError: Subclass must provide an implementation
        """
        raise NotImplementedError(
            "add method must be defined to use Noise base class")


class GaussianNoise(Noise):
    """Gaussian noise for images

    A gaussian noise is statistical noise having a probability density function equal to that of the gaussian distribution. 

    Principal sources of Gaussian noise in digital images arise during acquisition e.g. sensor noise caused by poor illumination 
    and/or high temperature, and/or transmission e.g. electronic circuit noise.

    """

    def __init__(self, mean=0, std=10):
        """Init

        Keyword Arguments:
            mean {int} -- Normal distribution mean parameter (default: {0})
            std {int} -- Normal distribution standard deviation parameter (default: {10})
        """
        self.mean = mean
        self.std = std

    def add(self, img):
        """Add gaussian noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive gaussian noise
        """
        w, h = img.shape
        gauss = np.random.normal(self.mean, self.std, (w, h)).astype(img.dtype)
        return cv2.add(img, gauss)


class PoissonNoise(Noise):
    """Poisson noise for images

    Poisson noise is a type of noise which can be modeled by a Poisson process. In electronics poisson noise originates from the discrete nature of electric charge. 
    Poisson noise also occurs in photon counting in optical devices, where shot noise is associated with the particle nature of light.
    """

    def __init__(self, factor=3):
        """Init

        Keyword Arguments:
            factor {int} -- Noise factor (default: {3})
        """

        self.factor = factor

    def add(self, img):
        """Add poisson noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive poisson noise
        """

        factor = 1 / self.factor
        img = np.array(img)
        img_noise = np.random.poisson(img * factor) / float(factor)
        np.clip(img_noise, 0, 255, img_noise)
        return img_noise


class UniformNoise(Noise):
    """Uniform noise for images

    Uniform noise results from the quantization of the pixels in an image. It has a uniform distribution but can be signal-dependent.
    """

    def __init__(self, amplitude=50):
        """Init

        Keyword Arguments:
            amplitude {int} -- Multiplicative amplitutde in ]0;255[ (default: {50})

        Raises:
            ValueError: Wrong amplitude given
        """
        if amplitude <= 0 or amplitude >= 255:
            raise ValueError("Amplitude must be in ]0; 255[")
        self.amplitude = amplitude

    def add(self, img):
        """Add uniform noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive uniform noise
        """
        uniform = np.random.rand(*img.shape) * self.amplitude
        return cv2.add(img, uniform.astype(img.dtype))


class SaltPepperNoise(Noise):
    """Salt and pepper noise for images

    This noise can be caused by sharp and sudden disturbances in the image signal. It presents itself as sparsely occurring white and black pixels.
    """

    def __init__(self, prob=0.05):
        """Init

        Keyword Arguments:
            prob {float} -- Salt and pepper probability (default: {0.05})

        Raises:
            ValueError: Wrong frequency given
            ValueError: Wrong probability given
        """
        if prob <= 0 or prob >= 1:
            raise ValueError("Salt or pepper probability must be in ]0; 1[")

        self.prob = prob

    def add(self, img):
        """Add salt and pepper noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Image with salt and pepper
        """

        salt_pepper = np.zeros(img.shape, np.float32)
        thres = 1 - self.prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < self.prob:
                    salt_pepper[i][j] = 0
                elif rdn > thres:
                    salt_pepper[i][j] = 255
                else:
                    salt_pepper[i][j] = img[i][j]
        return salt_pepper


class SquareMaskNoise(Noise):
    """Random rectangular masks

    This type of noise may be caused by a sensor failure or a stain on the lens. This type of noise is characterized by missing parts of the image.
    """

    def __init__(self, mask_shape, freq):
        """Init

        Arguments:
            mask_shape {tuple} -- Mask shape (width, heigt)
            freq {float} -- Frequency in ]0;1[

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        if len(mask_shape) != 2 or type(mask_shape) != tuple:
            raise ValueError("Mask must be a tuple of len == 2")
        if freq <= 0 or freq >= 1:
            raise ValueError("Frequency must be in ]0; 1[")

        self.mask_shape = mask_shape
        self.target_freq = freq

    def _compute_iter(self, img_shape):
        """Computes required iterations to achieve desired occurency

        Arguments:
            img_shape {tuple} -- Image's shape

        Returns:
            int --Number of required iterations
        """
        freq = (self.mask_shape[0] * self.mask_shape[1]
                ) / (img_shape[0] * img_shape[1])
        return int(self.target_freq / freq)

    def add(self, img):
        """Adds squarred masks to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- The image with masked areas
        """
        w, h = img.shape
        mask = np.ones((w, h), dtype=bool)
        for i in range(self._compute_iter((w, h))):
            x = np.random.randint(img.shape[0] - self.mask_shape[0])
            y = np.random.randint(img.shape[1] - self.mask_shape[1])
            off_x, off_y = x + self.mask_shape[0], y + self.mask_shape[1]
            mask[x:off_x, y:off_y] = False
        return np.where(mask, img, 0)


class SpeckleNoise(Noise):
    """Speckle noise

    Speckle is a form of multiplicative noise, which occurs when a pulse of a sound wave arbitrarily interferes 
    with small particles or objects on a scale comparable to the wavelength of the sound. 
    """

    def __init__(self, intensity=0.2):
        """Init

        Arguments:
            intensity {float} -- Noise intensity (default: {0.2})
        """
        self.intensity = intensity

    def add(self, img):
        """Adds speckle noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive speckle noise
        """

        gauss = np.random.normal(0,self.intensity ,img.size).reshape(100,100)
        return img + img * gauss
