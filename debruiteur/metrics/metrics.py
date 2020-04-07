from skimage import metrics
import cv2
import pandas as pd
from ..plots.plots import *


def compare_images(orignal_img, transformed_img):
    """Compares original image to transformed image

    Arguments:
        orignal_img {Array} -- Numpy like array of image
        transformed_img {Array} -- Numpy like array of image

    Returns:
        dict -- {"MSE", "NRMSE", "PSNR", "SSIM"}
    """
    mse = metrics.mean_squared_error(orignal_img, transformed_img)
    nrmse = metrics.normalized_root_mse(orignal_img, transformed_img)
    psnr = metrics.peak_signal_noise_ratio(orignal_img, transformed_img)
    ssim = metrics.structural_similarity(
        orignal_img, transformed_img, multichannel=True)

    return {"MSE": mse, "NRMSE": nrmse, "PSNR": psnr, "SSIM": ssim}


def metrics_example(dataframe, noise_class_list):
    """Show metrics example 

    Arguments:
        dataframe {Dataframe} -- Dataframe that contains images path
        noise_class_list {List} -- List of noise type

    Returns:
        Dataframe -- Dataframe that contain metrics example
    """
    path = dataframe.iloc[0, 0]
    orignal_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    images = [orignal_img]

    df_error = pd.DataFrame(
        {"Noise": [], "MSE": [], "NRMSE": [], "PSNR": [], "SSIM": []})

    for noise in noise_class_list:
        noised_img = noise.add(orignal_img)
        images.append(noised_img)

        noise_name = noise.__class__.__name__
        mse = metrics.mean_squared_error(orignal_img, noised_img)
        nrmse = metrics.normalized_root_mse(orignal_img, noised_img)
        psnr = metrics.peak_signal_noise_ratio(orignal_img, noised_img)
        ssim = metrics.structural_similarity(
            orignal_img, noised_img, multichannel=True)

        df_error = df_error.append(
            {"Noise": noise_name, "MSE": mse, "NRMSE": nrmse, "PSNR": psnr, "SSIM": ssim}, ignore_index=True)

    plot_im_grid_from_list(images, 5, 2)
    df_error.head(len(noise_class_list))
    return df_error
