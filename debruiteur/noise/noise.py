from abc import ABC, abstractmethod
import numpy as np
import cv2

class Noise(ABC):

    @abstractmethod
    def add(self, img):
        raise NotImplementedError(
            "add method must be defined to use Noise base class")


class GaussianNoise(Noise):

    def __init__(self, mean=0, std=10):
        self.mean = mean
        self.std = std

    def add(self, img):
        w, h, c = img.shape
        gauss = np.random.normal(
            self.mean, self.std, (w, h, c)).astype(img.dtype)
        return cv2.add(img, gauss)


class PoissonNoise(Noise):

    def add(self, img):
        poisson = np.random.poisson(img).astype(img.dtype)
        return cv2.add(img, poisson)


class UniformNoise(Noise):

    def __init__(self, amplitude=50):
        if 0 <= amplitude >= 255:
            raise ValueError("Amplitude must be in ]0; 255[")
        self.amplitude = amplitude

    def add(self, img):
        uniform = np.random.rand(*img.shape) * self.amplitude
        return cv2.add(img, uniform.astype(img.dtype))


class SaltPepperNoise(Noise):

    def __init__(self, p=0.1):
        if 0 <= p >= 1:
            raise ValueError("Probability must be in ]0; 1[")
        self.p = p

    def add(self, img):
        w, h, _ = img.shape
        mask = np.random.rand(w, h) > self.p
        mask3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return np.where(mask3d, img, 0 if np.random.rand(1) > 0.5 else 1)


class SquareMaskNoise(Noise):

    def __init__(self, mask_shape, freq):
        if len(mask_shape) != 2:
            raise ValueError("Mask must have 2 dimensions")
        if 0 <= freq >= 1:
            raise ValueError("Frequency must be in ]0; 1[")
        self.mask_shape = mask_shape
        self.target_freq = freq

    def _compute_iter(self, img_shape):
        freq = (self.mask_shape[0] * self.mask_shape[1]
                ) / (img_shape[0] * img_shape[1])
        return int(self.target_freq / freq)

    def add(self, img):
        w, h, c = img.shape
        mask = np.ones((w, h, c), dtype=bool)
        for i in range(self._compute_iter((w, h))):
            x = np.random.randint(img.shape[0] - self.mask_shape[0])
            y = np.random.randint(img.shape[1] - self.mask_shape[1])
            off_x, off_y = x + self.mask_shape[0], y + self.mask_shape[1]
            mask[x:off_x, y:off_y, :] = False
        return np.where(mask, img, 0)


class SpeckleNoise(Noise):

    def add(self, img):
        normal = np.random.randn(*img.shape).astype(img.dtype)
        return cv2.add(img, cv2.multiply(img, normal))
