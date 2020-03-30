"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import random

from ..utils.utils import init_dir
from ..noise.noise import *


def make_original_dataframe(base_path="images"):
    """Makes a dataframe from an image directory

    Keyword Arguments:
        base_path {str} -- Image directory's path (default: {"images"})

    Returns:
        DataFrame -- Columns : [path : image path, size_x: image width, size_y: image height]
    """
    np.random.seed(42)

    images_folder = np.random.choice(os.listdir(base_path), 20)

    images_path = []
    images_size = []

    for folder in images_folder:
        image_folder = os.path.join(base_path, folder)

        for img_name in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_name).replace("\\", "/")
            images_path.append(img_path)
            img = Image.open(img_path)
            images_size.append(img.size)

    dataframe = pd.DataFrame(pd.Series(images_path).to_frame(
        "path")).join(pd.Series(images_size).to_frame("size"))

    dataframe[["size_x", "size_y"]] = pd.DataFrame(
        dataframe["size"].tolist(), index=dataframe.index)
    dataframe.drop(["size"], axis=1, inplace=True)

    return dataframe[(dataframe["size_x"] > 100) & (dataframe["size_y"] > 100)]


def crop_img(img, shape=(100, 100)):
    """Crops image from center

    Arguments:
        img {Array} -- Numpy like array containing the image

    Keyword Arguments:
        shape {tuple} -- Crop (default: {(100, 100)})

    Returns:
        Array -- The cropped image
    """
    width, height = img.shape

    cx, cy = width / 2, height / 2
    sx, sy = cx - shape[0] / 2, cy - shape[1] / 2
    ex, ey = cx + shape[0] / 2, cy + shape[1] / 2

    return img[int(sx): int(ex), int(sy): int(ey)]


def make_resized_dataframe(dataframe, img_shape=(100, 100), resized_path="resized_images"):
    """Preprocesses all the images contained in the dataframe

    Arguments:
        dataframe {DataFrame} -- DataFrame containing columns [path, image width, image height]

    Keyword Arguments:
        resized_path {str} -- Preprocessed image directory (default: {"resized_images"})

    Returns:
        DataFrame -- DataFrame with preprocessed image's path
    """
    init_dir(resized_path)

    image_names = []

    for index, row in tqdm(dataframe.iterrows()):
        path, size_x, size_y = row[0], row[1], row[2]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        croped_img = crop_img(img, img_shape)

        path = os.path.join(resized_path, f"img{index}.jpg")

        image_names.append(path)
        cv2.imwrite(path, croped_img)

    return pd.DataFrame({'path': image_names})

def make_noised_dataframe(dataframe, noise_list, noised_path="noised_images"):
    """Add noise to all images in the dataframes

    Arguments:
        dataframe {DataFrame} -- Dataframe with resized image path
        noise_list {list} -- List of Noise's sublclass instance

    Keyword Arguments:
        noised_path {str} -- path to the output directory (default: {"noised_images"})

    Raises:
        ValueError: Invalid noises

    Returns:
        DataFrame -- Dataframe with noised image's path
    """
    original_image_names = []
    noised_image_names = []

    init_dir(noised_path)

    for index, row in tqdm(dataframe.iterrows()):
        rand_noise = random.choice(noise_list)

        if not issubclass(type(rand_noise), Noise) and rand_noise is not None:
            raise ValueError("noise is not of valid type Noise")

        img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)

        noised_img = rand_noise.add(img)
        noised_class_name = type(rand_noise).__name__
        path = os.path.join(noised_path, f"img{index}_{noised_class_name}.jpg")

        cv2.imwrite(path, noised_img)
        original_image_names.append(row['path'])
        noised_image_names.append(path)
    return pd.DataFrame({'original_path': original_image_names, 'noised_path': noised_image_names})
