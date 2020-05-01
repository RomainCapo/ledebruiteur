"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.util import img_as_float
from skimage.restoration import wiener, unsupervised_wiener


def wiener_filter(img, unsupervised=True, wiener_balance=1100, psf_size=5, psf_numerator=25):
    """Wiener filter to sharpen an image

    This filter is used to estimate the desired value of a noisy signal.
    The Wiener filter minimizes the root mean square error between the estimated random process and the desired process.

    Arguments:
        img {array} -- Image array [Non-normalize (0-255)]

    Keyword Arguments:
        unsupervised {bool} -- true for supervised algorithm, false otherwise (default: {True})
        wiener_balance {int} -- Wiener balance parameter (default: {1100})
        psf_size {int} -- PSF kernel size (default: {5})
        psf_numerator {int} -- PSF kernel numerator (default: {25})

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
    """

    img = np.array(img, np.float32)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_32F)  # Allow to normalize image

    psf = np.ones((psf_size, psf_size)) / psf_numerator
    convolved_img = convolve2d(img, psf, 'same')
    convolved_img += 0.1 * convolved_img.std() * \
        np.random.standard_normal(convolved_img.shape)

    deconvolved = None
    if unsupervised:
        deconvolved, _ = unsupervised_wiener(convolved_img, psf)
    else:
        deconvolved = wiener(convolved_img, psf, wiener_balance)

    cv2.absdiff(deconvolved, deconvolved)

    return deconvolved * 255


def laplacian_filter(img, gaussian_kernel_size=5):
    """Use Laplacian to filter to sharpen an image

    This filter calculates the Laplace transform on an image and adds it to the original image.
    Gaussian noise can be added to the image to improve the result.

    Arguments:
        img {array} -- Image source array [Non-normalize (0-255)]

    Keyword Arguments:
        gaussian_kernel_size {int} -- size of the gaussian blur kernel, if None the gaussian blur kernel is not applied (default: {None})

    Returns:
        array -- Filterd image [Non-normalize (0-255)]
    """

    img = np.array(img, np.float32)

    if gaussian_kernel_size is not None:
        img = cv2.GaussianBlur(
            img, (gaussian_kernel_size, gaussian_kernel_size), 0)

    laplace_img = cv2.Laplacian(img, cv2.CV_32F)
    return img + laplace_img

def gaussian_filter(img, gaussian_kernel_size=(3, 3), sigma_x=3):
    """Use gaussian blur to reduce noise on an image

    The gaussian filter is used to reduce noise by convolving a gaussian kernel.
    It is particularly good with gaussian noise.

    Arguments:
        img {array} -- Image source array [Non-normalize (0-255)]

    Keyword Arguments:
        gaussian_kernel_size {tuple or int} -- kernel size of the gaussian kernel, if (0,0) the kernel is define with the sigma value (default: {(0,0)})
        sigma_x {int} -- Sigma X value of the gaussian kernel (default: {3})

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
    """
    return cv2.GaussianBlur(img, gaussian_kernel_size, sigma_x)


def gaussian_weighted_substract_filter(img, gaussian_kernel_size=(0, 0), sigma_x=3, weighted_alpha=1.5, weighted_beta=-0.5, weighted_gamma=0):
    """Use gaussian filter to sharpen an image

    This filter calculates a Gaussian blur on the image and adds it with certain weightings configurable with the parameters of the function.

    Arguments:
        img {array} -- Image source array [Non-normalize (0-255)]

    Keyword Arguments:
        gaussian_kernel_size {tuple or int} -- kernel size of the gaussian kernel, if (0,0) the kernel is define with the sigma value (default: {(0,0)})
        sigma_x {int} -- Sigma X value of the gaussian kernel (default: {3})
        weighted_alpha {float} -- Alpha parameter of weighted function (default: {1.5})
        weighted_beta {float} -- Beta parameter of weighted function (default: {-0.5})
        weighted_gamma {int} -- Gamma parameter of weighted function (default: {0})

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
    """
    img = np.array(img, np.float32)

    gaussian_img = cv2.GaussianBlur(img, gaussian_kernel_size, sigma_x)
    return cv2.addWeighted(img, weighted_alpha, gaussian_img, weighted_beta, weighted_gamma)


def mean_filter(img, kernel_size=5):
    """Mean filter for noise reduction

    The averaging filter replaces each pixel with the value average of its neighbours according to a given kernel. 

    Arguments:
        img {array} -- Image source array [Non-normalize (0-255)]
        kernel_size {int} -- Kernel size (default: {5})

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
    """

    img = np.array(img, np.float32)

    return cv2.boxFilter(img, cv2.CV_32F, (kernel_size, kernel_size))


def median_filter(img, kernel_size=5):
    """Median filter for noise reduction

    The median filter replaces each pixel with the median value of its neighbors according to a given kernel. 

    Arguments:
        img {array} -- Image source array [Non-normalize (0-255)]

    Keyword Arguments:
        kernel_size {int} -- Kernel size (default: {5})

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
    """

    img = np.array(img, np.float32)

    return cv2.medianBlur(img, kernel_size)


def conservative_filter(img, filter_size=5):
    """Conservative filter for image noise reduction
    Code from : https://towardsdatascience.com/image-filters-in-python-26ee938e57d2

    The conservative filter for a given kernel retains the current pixel if it is between the min and max of neighboring pixels. 
    If the value of the pixel is below the min of the neighboring pixels have taken the min. 
    If the value is above the neighbouring max pixel, the max is taken.

    Arguments:
        img {array} -- Image source arary [Non-normalize (0-255)]
        filter_size {int} -- Kernel size

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
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
    return new_image

def dft(pass_filter):
    """Digital fourier transform

    Arguments:
        pass_filter {function} -- A function returning a mask
    """
    def inner(img):
        img = np.array(img, np.float32)
        dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        dft_shift = np.fft.fftshift(dft)

        mask = pass_filter(img)

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        result_img = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
        result_img = cv2.magnitude(result_img[:, :, 0], result_img[:, :, 1])

        return result_img
    return inner

@dft
def low_pass_filter(img):
    """Image noise reduction with Digital Fourier Transform

    This low pass filter performs a Fourier transform on the image.
    Afterwards, a mask is applied on the image that has undergone the Fourier transform and the image is retransformed with an inverse Fourier transform.

    Arguments:
        img {array} -- Image source array [Non-normalize (0-255)]

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
    """
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    return mask

@dft
def high_pass_filter(img):
    """Image noise reduction with Digital Fourier Transform

    This high pass filter performs a Fourier transform on the image.
    Afterwards, a mask is applied on the image that has undergone the Fourier transform and the image is retransformed with an inverse Fourier transform.

    Arguments:
        img {array} -- Image source array [Non-normalize (0-255)]

    Returns:
        array -- Filtered image [Non-normalize (0-255)]
    """
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-1:crow+1, ccol-1:ccol+1] = 0

    return mask

def gaussian_blur(img, kernel=(5,5)):
    """Adds gaussian blur noise

    Arguments:
        img {Array} -- Numpy like array of image

    Keyword Arguments:
        kernel {tuple} -- kernel shape of the blur

    Returns:
        Array -- Gaussian blur noise
    """
    return cv2.GaussianBlur(img, kernel, 0)

def averaging_blur(img, kernel=(5,5)):
    """Adds averaging blur noise

    Arguments:
        img {Array} -- Numpy like array of image

    Keyword Arguments:
        kernel {tuple} -- kernel shape of the blur

    Returns:
        Array -- Gaussian blur noise
    """
    return cv2.blur(img, kernel)

def median_blur(img, k_size=5):
    """Adds median blur noise

    Arguments:
        img {Array} -- Numpy like array of image

    Keyword Arguments:
        k_size {int} -- size of the blur

    Returns:
        Array -- Gaussian blur noise
    """
    return cv2.medianBlur(img, k_size)