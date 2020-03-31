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
from tqdm.notebook import tqdm
import statistics
from ..metrics.metrics import *


def compute_noise_reduction_method_statistics(df_images, noise_reduction_methods, img_size=64):
    """Compute the score of each filter for each metrics

    Arguments:
        df_images {Dataframe} -- Dataframe that containt two columns, first with original image path, the second with noised image path 
        noise_reduction_methods {list} -- List of tuple, each tuple contain the name of the noise reduction method and the noise reduction function

    Keyword Arguments:
        img_size {int} -- size of the image, assume the image are square (default: {100})

    Returns:
        Dataframe -- Dataframe containing on each column the score with a metric for each noise reduction method
    """

    df_stats = pd.DataFrame(columns=['MSE', 'NRMSE', 'PSNR', 'SSIM'])
    for name, method in noise_reduction_methods:
        mse_values = []
        nrmse_values = []
        psnr_values = []
        ssmi_values = []

        for index, row in tqdm(df_images.iterrows()):
            noised_img = cv2.imread(row['original_path'], cv2.IMREAD_GRAYSCALE)
            original_img = cv2.imread(row['noised_path'], cv2.IMREAD_GRAYSCALE)
            new_img = method(noised_img)

            if new_img.shape == (img_size, img_size):
                scores = compare_images(original_img, new_img)
            elif new_img.shape == (1, img_size, img_size, 1):
                scores = compare_images(
                    original_img, new_img.reshape(img_size, img_size))

            mse_values.append(scores['MSE'])
            nrmse_values.append(scores['NRMSE'])
            psnr_values.append(scores['PSNR'])
            ssmi_values.append(scores['SSIM'])

        df_stats.loc[name] = [statistics.mean(mse_values), statistics.mean(
            nrmse_values), statistics.mean(psnr_values), statistics.mean(ssmi_values)]
        print(f"Compute finish for {name}")
    return df_stats


def compute_noise_type_statistics(df_image, noise_reduction_methods, noise_type, metrics='MSE'):
    """Compute the score for each reduction method for each noise type

    Arguments:
        df_image {Dataframe} -- Dataframe with one column that contains the original image path 
        noise_reduction_methods {list} -- List of tuple, each tuple contain the name of the noise reduction method and the noise reduction function
        noise_type {list} -- List that contain each noise type object

    Keyword Arguments:
        metrics {str} -- Score metrics, Allowed : [MSE, NRMSE, PSNR, SSMI] (default: {'MSE'})

    Returns:
        Dataframe -- Dataframe containing the score of each noise reduction method for each noise type
    """

    df_stats = pd.DataFrame(
        columns=[type(noise).__name__ for noise in noise_type])

    for name, method in noise_reduction_methods:

        noise_values_list = []

        for noise in tqdm(noise_type):

            values_list = []

            for index, row in df_image.iterrows():
                img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
                noised_img = noise.add(img)
                processed_img = method(noised_img)
                scores = compare_images(img, processed_img)
                values_list.append(scores[metrics])

            noise_values_list.append(statistics.mean(values_list))

        df_stats.loc[name] = noise_values_list
        print(f"Compute finish for {name}")

    return df_stats
