"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

from skimage.util import img_as_float
from scipy.signal import convolve2d
from skimage.restoration import wiener, unsupervised_wiener
import numpy as np
import cv2


def wiener_filter(img, unsupervised=True, wiener_balance=1100, psf_size=5, psf_numerator=25):
    """Wiener filter on a image

    Arguments:
        img {array} -- Image array

    Keyword Arguments:
        unsupervised {bool} -- true for supervised algorithm, false otherwise (default: {True})
        wiener_balance {int} -- Wiener balance parameter (default: {1100})
        psf_size {int} -- PSF kernel size (default: {5})
        psf_numerator {int} -- PSF kernel numerator (default: {25})

    Returns:
        array -- Filtered image
    """
    img = img_as_float(img)

    psf = np.ones((psf_size, psf_size)) / psf_numerator
    convolved_img = convolve2d(img, psf, 'same')
    convolved_img += 0.1 * convolved_img.std() * \
        np.random.standard_normal(convolved_img.shape)

    deconvolved = None
    if unsupervised:
        deconvolved, _ = unsupervised_wiener(convolved_img, psf)
    else:
        deconvolved = wiener(convolved_img, psf, wiener_balance)

    return deconvolved


def laplacian_filter(img, gaussian_kernel_size=None):
    """Use Laplacian to reduce noise on a image

    Arguments:
        img {array} -- Image source array

    Keyword Arguments:
        gaussian_kernel_size {int} -- size of the gaussian blur kernel, if None the gaussian blur kernel is not applied (default: {None})

    Returns:
        array -- Filterd image
    """
    if gaussian_kernel_size is not None:
        img = cv2.GaussianBlur(
            img, (gaussian_kernel_size, gaussian_kernel_size), 0)

    laplace_img = cv2.Laplacian(img, cv2.CV_64F)
    return img + laplace_img


def gaussian_weighted_substract_filter(img, gaussian_kernel_size=(0, 0), sigma_x=3, weighted_alpha=1.5, weighted_beta=-0.5, weighted_gamma=0):
    """Use gaussian filter to reduce noise on a image

    Arguments:
        img {array} -- Image source array

    Keyword Arguments:
        gaussian_kernel_size {tuple or int} -- kernel size of the gaussian kernel, if (0,0) the kernel is define with the sigma value (default: {(0,0)})
        sigma_x {int} -- Sigma X value of the gaussian kernel (default: {3})
        weighted_alpha {float} -- Alpha parameter of weighted function (default: {1.5})
        weighted_beta {float} -- Beta parameter of weighted function (default: {-0.5})
        weighted_gamma {int} -- Gamma parameter of weighted function (default: {0})

    Returns:
        array -- Filtered image
    """

    gaussian_img = cv2.GaussianBlur(img, gaussian_kernel_size, sigma_x)
    return cv2.addWeighted(img, weighted_alpha, gaussian_img, weighted_beta, weighted_gamma)


def mean_filter(img, kernel_size=5):
    """Mean filter for noise reduction

    Arguments:
        img {array} -- Image source array
        kernel_size {int} -- Kernel size (default: {5})

    Returns:
        array -- Filtered image
    """
    return cv2.boxFilter(img, cv2.CV_64F, (kernel_size, kernel_size))


def median_filter(img, kernel_size=5):
    """Median filter for noise reduction

    Arguments:
        img {array} -- Image source array

    Keyword Arguments:
        kernel_size {int} -- Kernel size (default: {5})

    Returns:
        array -- Filtered image
    """
    return cv2.medianBlur(img, kernel_size)


def conservative_filter(img, filter_size):
    """Conservative filter for image noise reduction
    Code from : https://towardsdatascience.com/image-filters-in-python-26ee938e57d2

    Arguments:
        img {array} -- Image source arary
        filter_size {int} -- Kernel size

    Returns:
        array -- Filtered image
    """
    temp = []

    indexer = filter_size // 2
    new_image = img.copy()
    nrow, ncol = img.shape

    for i in range(nrow):
        for j in range(ncol):
            for k in range(i-indexer, i+indexer+1):
                for m in range(j-indexer, j+indexer+1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(img[k, m])
            temp.remove(img[i, j])

            max_value = max(temp)
            min_value = min(temp)

            if img[i, j] > max_value:
                new_image[i, j] = max_value

            elif img[i, j] < min_value:
                new_image[i, j] = min_value

            temp = []
    return new_image.copy()


def fft_filter(img, mask=5):
    """Image noise reduction with Digital Fourier Transform

    Arguments:
        img {array} -- Image source array

    Keyword Arguments:
        mask {int} -- mask size  (default: {5})

    Returns:
        array -- Filtered image
    """

    img_float = np.float32(img)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    fshift = dft_shift*mask

    f_ishift = np.fft.ifftshift(fshift)

    new_img = cv2.idft(f_ishift)
    return cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])
