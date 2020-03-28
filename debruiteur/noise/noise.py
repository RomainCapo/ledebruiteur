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
        w, h, c = img.shape
        gauss = np.random.normal(
            self.mean, self.std, (w, h, c)).astype(img.dtype)
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
        if 0 <= amplitude >= 255:
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

    def __init__(self, p=0.1):
        """Init

        Keyword Arguments:
            p {float} -- Probability (default: {0.1})

        Raises:
            ValueError: Wrong probability given
        """
        if 0 <= p >= 1:
            raise ValueError("Probability must be in ]0; 1[")
        self.p = p

    def add(self, img):
        """Add salt and pepper noise to the image

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Image with salt and pepper
        """
        w, h, _ = img.shape
        mask = np.random.rand(w, h) > self.p
        mask3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return np.where(mask3d, img, 0 if np.random.rand(1) > 0.5 else 1)


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
        if len(mask_shape) != 2:
            raise ValueError("Mask must have 2 dimensions")
        if 0 <= freq >= 1:
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
        w, h, c = img.shape
        mask = np.ones((w, h, c), dtype=bool)
        for i in range(self._compute_iter((w, h))):
            x = np.random.randint(img.shape[0] - self.mask_shape[0])
            y = np.random.randint(img.shape[1] - self.mask_shape[1])
            off_x, off_y = x + self.mask_shape[0], y + self.mask_shape[1]
            mask[x:off_x, y:off_y, :] = False
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

    def __init__(self, kernel=(5,5)):
        """Init

        Keyword Arguments:
            kernel {tuple} -- kernel of the blur
        """
        self.kernel = kernel

    def add(self, img):
        """Adds averaging blur noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive speckle noise
        """
        return cv2.blur(img, self.kernel)

class GaussianBlurNoise(Noise):
    """Gaussian blur noise"""

    def __init__(self, kernel=(5,5)):
        """Init

        Keyword Arguments:
            kernel {tuple} -- kernel of the blur
        """
        self.kernel = kernel

    def add(self, img):
        """Adds averaging blur noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive speckle noise
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
        """Adds averaging blur noise

        Arguments:
            img {Array} -- Numpy like array of image

        Returns:
            Array -- Additive speckle noise
        """
        return cv2.medianBlur(img, self.ksize)