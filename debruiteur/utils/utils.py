"""
Le Debruiteur
Jonas Freiburghaus
Romain Capocasale
He-Arc, INF3dlm-a
Image Processing course
2019-2020
"""

import os

import shutil
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model as k_load_model


def init_dir(directory, delete_existing=True):
    """Creates a directory if not exists and remove content if it exists

    Arguments:
        directory {string} -- path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if delete_existing:
            shutil.rmtree(directory)
            os.makedirs(directory)


def split_train_val_df(df, p=0.8):
    """Splits DataFrame in train df and validation df

    Arguments:
        df {DataFrame} -- Original DataFrame

    Keyword Arguments:
        p {float} -- Probability in ]0:1[ (default: {0.8})

    Raises:
        ValueError: Must be a panda's Dataframe
        ValueError: Probability must be in ]0;1[

    Returns:
        [type] -- [description]
    """
    if type(df) != pd.DataFrame:
        raise ValueError("Df must be panda's DataFrame")
    if 0 <= p >= 1:
        raise ValueError("p must be a probability in ]0;1[")

    msk = np.random.rand(len(df)) < p
    return df[msk], df[~msk]


def save_model(model, path, name):
    """Saves a keras model

    Arguments:
        model {Model} -- Keras model
        path {String} -- Save directory
        name {name} -- File name
    """
    init_dir(path, delete_existing=False)
    model.save(os.path.join(path, name))


def load_model(path, name):
    """Loads a keras model

    Arguments:
        path {String} -- Load directory
        name {String} -- File name

    Returns:
        Model -- Keras model
    """
    return k_load_model(os.path.join(path, name))
