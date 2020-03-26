import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm

def make_dataframe(base_path="images"):
    np.random.seed(42)

    images_folder = np.random.choice(os.listdir(base_path), 20)

    images_path = []
    images_size = []
        
    for folder in images_folder:
        image_folder = os.path.join(base_path, folder)
        
        for img_name in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_name).replace("\\","/")
            images_path.append(img_path)
            img = Image.open(img_path)
            images_size.append(img.size)
            
    dataframe = pd.DataFrame(pd.Series(images_path).to_frame("path")).join(pd.Series(images_size).to_frame("size"))

    dataframe['size_x'], dataframe['size_y'] = list(dataframe['size'].str)
    dataframe.drop(['size'], axis=1, inplace=True)

    return dataframe[(dataframe['size_x'] > 100) & (dataframe['size_y'] > 100)]

def crop_img(img, shape=(100,100)):
    width, height, _ = img.shape
    
    cx, cy = width / 2, height /2
    sx, sy = cx - shape[0] / 2, cy - shape[1] / 2
    ex, ey = cx + shape[0] / 2, cy + shape[1] / 2

    return img[int(sx) : int(ex), int(sy) : int(ey)]

def preprocess(dataframe, resized_path="resized_images"):
    image_names = []

    for index, row in tqdm(dataframe.iterrows()):
        path, size_x, size_y = row['path'], row['size_x'], row['size_y']
        img = cv2.imread(path)
        croped_img = crop_img(img)
        
        path = os.path.join(resized_path, f"img{index}.jpg")
        
        image_names.append(path)
        cv2.imwrite(path, croped_img)

    return image_names