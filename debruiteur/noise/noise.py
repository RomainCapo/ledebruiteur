"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2


class Noise(ABC):

    @abstractmethod
    def add(self, img):
        """Add noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError(
            "add method must be defined to use Noise base class")


class GaussianNoise(Noise):
    """Gaussian noise for images"""

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
    """Poisson noise for images"""

    def add(self, img):
        """Add poisson noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive poisson noise
        """
        poisson = np.random.poisson(img).astype(img.dtype)
        return cv2.add(img, poisson)


class UniformNoise(Noise):
    """Uniform noise for images"""

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
    """Salt and pepper noise for images"""

    def __init__(self, freq=0.1, s_or_p_prob=0.5):
        """Init

        Keyword Arguments:
            freq {float} -- Frequency (default: {0.1})
            s_and_p_prob {float} -- Salt and pepper probability (default: {0.5})

        Raises:
            ValueError: Wrong frequency given
            ValueError: Wrong probability given
        """
        if freq <= 0 or freq >= 1:
            raise ValueError("Frequency must be in ]0; 1[")
        if s_or_p_prob <= 0 or s_or_p_prob >= 1:
            raise ValueError("Salt or pepper probability must be in ]0; 1[")

        self.freq = freq
        self.s_or_p_prob = s_or_p_prob

    def add(self, img):
        """Add salt and pepper noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Image with salt and pepper
        """
        w, h = img.shape
        mask = np.random.rand(w, h) > self.freq

        return np.where(mask, img, 0 if np.random.rand(1) > self.s_or_p_prob else 1)


class SquareMaskNoise(Noise):
    """Random rectangular masks"""

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
    """Speckle noise"""

    def add(self, img):
        """Adds speckle noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive speckle noise
        """
        normal = np.random.randn(*img.shape).astype(img.dtype)
        return cv2.add(img, cv2.multiply(img, normal))


class AveragingBlurNoise(Noise):
    """Averaging blur noise"""

    def __init__(self, kernel=(5, 5)):
        """Init

        Keyword Arguments:
            kernel {tuple} -- kernel shape of the blur
        """
        self.kernel = kernel

    def add(self, img):
        """Adds averaging blur noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Averaging blur noise
        """
        return cv2.blur(img, self.kernel)


class GaussianBlurNoise(Noise):
    """Gaussian blur noise"""

    def __init__(self, kernel=(5, 5)):
        """Init

        Keyword Arguments:
            kernel {tuple} -- kernel shape of the blur
        """
        self.kernel = kernel

    def add(self, img):
        """Adds gaussian blur noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Gaussian blur noise
        """
        return cv2.GaussianBlur(img, self.kernel, 0)


class MedianBlurNoise(Noise):
    """Median blur noise"""

    def __init__(self, ksize=5):
        """Init

        Keyword Arguments:
            ksize {int} -- size of the blur
        """
        self.ksize = ksize

    def add(self, img):
        """Adds median blur noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Median blur noise
        """
        return cv2.medianBlur(img, self.ksize)
