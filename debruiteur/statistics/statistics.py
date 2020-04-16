"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import cv2
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from debruiteur.metrics.metrics import compare_images


def compute_noise_reduction_method_statistics(dg_images, noise_reduction_methods, img_size=100, verbose=True):
    """Compute the score of each filter for each metrics

    Arguments:
        dg_images {DataGenrator} -- DataGenerator that contain the original images and the noised_image
        noise_reduction_methods {list} -- List of tuple, each tuple contain the name of the noise reduction method and the noise reduction function

    Keyword Arguments:
        img_size {int} -- size of the image, assume the image are square (default: {100})
        verbose {bool} -- display the progression of the statistics (default: {True})

    Returns:
        Dataframe -- Dataframe containing on each column the score with a metric for each noise reduction method
    """
    df_stats = pd.DataFrame(columns=['MSE', 'NRMSE', 'PSNR', 'SSIM'])

    noised_images, original_images = dg_images[0]

    for name, method in noise_reduction_methods:
        mse_values = []
        nrmse_values = []
        psnr_values = []
        ssmi_values = []

        for x, y in zip(noised_images, original_images):
            y_pred = method(x.reshape(img_size, img_size).copy()
                            ).reshape(img_size, img_size)

            y = y.reshape(img_size, img_size)*255
            scores = compare_images(y, y_pred)

            mse_values.append(scores['MSE'])
            nrmse_values.append(scores['NRMSE'])
            psnr_values.append(scores['PSNR'])
            ssmi_values.append(scores['SSIM'])

        df_stats.loc[name] = [statistics.mean(mse_values), statistics.mean(
            nrmse_values), statistics.mean(psnr_values), statistics.mean(ssmi_values)]

        if verbose:
            print(f"Compute finish for {name}")
    return df_stats


def compute_noise_type_statistics(dg_images, noise_reduction_methods, noise_type, metrics='MSE', img_size=100, verbose=True):
    """Compute the score for each reduction method for each noise type

    Arguments:
        dg_images {DataGenrator} -- DataGenerator that contain the original images and the noised_image
        noise_reduction_methods {list} -- List of tuple, each tuple contain the name of the noise reduction method and the noise reduction function
        noise_type {list} -- List that contain each noise type object

    Keyword Arguments:
        metrics {str} -- Score metrics, Allowed : [MSE, NRMSE, PSNR, SSMI] (default: {'MSE'})
        img_size {int} -- size of the image, assume the image are square (default: {100})
        verbose {bool} -- display the progression of the statistics (default: {True})

    Returns:
        Dataframe -- Dataframe containing the score of each noise reduction method for each noise type
    """

    df_stats = pd.DataFrame(
        columns=[type(noise).__name__ for noise in noise_type])

    _, original_images = dg_images[0]

    for name, method in noise_reduction_methods:

        noise_values_list = []

        for noise in noise_type:

            values_list = []

            for img in original_images:
                img = img.reshape(img_size, img_size) * 255
                noised_img = noise.add(img)
                # Allow to normalize image
                noised_img = cv2.normalize(
                    noised_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                processed_img = method(noised_img.reshape(
                    img_size, img_size).copy()).reshape(img_size, img_size)
                scores = compare_images(img, processed_img)

                values_list.append(round(scores[metrics], 5))

            noise_values_list.append(statistics.mean(values_list))

        df_stats.loc[name] = noise_values_list
        print(f"Compute finish for {name}")

    return df_stats
