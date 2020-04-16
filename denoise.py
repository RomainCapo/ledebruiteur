import argparse
import cv2
import logging
import numpy as np
import os

from debruiteur.models.gan import *
from debruiteur.noise.filters import *
from debruiteur.preprocessing.preprocessor import crop_img
from debruiteur.utils.utils import load_model


MODELS = {
    'GAN': 'gan_generator_100epochs.h5',
    'CONV_AUTOENCODER': 'conv_autoencoder.h5',
    'DENSE_AUTOENCODER': 'dense_autoencoder.h5'
}

POST_PROCESSING = {
    'CONSERVATIVE_FILTER': conservative_filter,
    'FAST_FOURIER_TRANSFORM_FILTER': fft_filter,
    'GAUSSIAN_SUBSTRACT': gaussian_weighted_substract_filter,
    'LAPLACIAN_FILTER': laplacian_filter,
    'MEAN_FILTER': mean_filter,
    'MEDIAN_FILTER': mean_filter,
    'NO_FILTER': None,
    'WIENER_FILTER': wiener_filter
}


def denoise_img(in_img_path, out_img_path, model, img_filter):
    """Denoises a grayscale image

    Arguments:
        in_img_path {str} -- Input image path
        out_img_path {str} -- Output image path
        model {str} -- Model name
        img_filter {str} -- Filter name
    """
    logging.info('Reading image')
    img = cv2.imread(in_img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception('Image not found')

    logging.info('Croping image')
    img = crop_img(img)

    logging.info('Normalizing image')
    img = np.array(img, np.float32) / 255

    logging.info('Reshaping image')
    img = img.reshape(-1, 100, 100, 1)

    logging.info('Load model')
    model = open_model(model)

    logging.info('Denoising image')
    denoised_img = model.predict(img)

    logging.info('Reshaping denoised image')
    denoised_img = denoised_img.reshape(100, 100)

    logging.info('Denormalizing image')
    denoised_img *= 255

    filter_fn = open_filter(img_filter)
    if filter_fn:
        logging.info('Post processing image')
        denoised_img = filter_fn(denoised_img)

    logging.info('Saving image')
    cv2.imwrite(out_img_path, denoised_img)
    logging.info('Denoised image saved')


def open_model(model_name):
    """Opens the model

    Arguments:
        model_name {str} -- Model name

    Returns:
        Model -- Tensorflow model
    """
    MODEL_DIR = 'saved_models'
    MODEL_FILE = MODELS[model_name]

    return load_model(MODEL_DIR, MODEL_FILE)


def open_filter(filter_name):
    """Opens the filter

    Arguments:
        filter_name {str} -- Filter name

    Returns:
        function -- Filter function
    """
    return POST_PROCESSING[filter_name]


def parse_cli_args():
    """Parses the cli arguments"""
    parser = argparse.ArgumentParser(description='Denoise an image.')

    parser.add_argument('-i', '--image', required=True,
                        type=str, help="Input image path")

    parser.add_argument('-o', '--output_image', required=True,
                        type=str, help="Output image path")

    parser.add_argument(
        '-m',
        '--model',
        required=False,
        type=int,
        choices=range(len(MODELS)),
        default=0,
        help="""Model : \n
                [0] -> Generatrive adversial network \n
                [1] -> Autoencoder \n
                [2] -> Dense \n"""
    )

    parser.add_argument(
        '-p',
        '--post_processing',
        required=False,
        type=int,
        choices=range(len(POST_PROCESSING)),
        default=6,
        help="""Post processing algorithm (filter) : \n
                [0] -> Conservative \n
                [1] -> Fast Fourier transform \n
                [2] -> Gaussian weighted substract \n,
                [3] -> Laplacian \n,
                [4] -> Mean \n,
                [5] -> Median \n,
                [6] -> No filter \n
                [7] -> Wiener \n,
                """
    )

    args = parser.parse_args()
    return args.image, args.output_image, list(MODELS.keys())[args.model], list(POST_PROCESSING.keys())[args.post_processing]


if __name__ == '__main__':
    in_img_path, out_img_path, model, img_filter = parse_cli_args()

    try:
        denoise_img(in_img_path, out_img_path, model, img_filter)
    except BaseException as e:
        logging.error(e)
