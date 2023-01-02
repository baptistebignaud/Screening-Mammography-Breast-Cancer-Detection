import random
from typing import List
import numpy as np
import warnings
from PIL import Image
import os
import cv2
import pandas as pd

img_path = "../kaggle_dataset"


def load_file(
    files: str or List[str] or List[np.array], n: int, m: int = None, seed: int = None
) -> List:
    """
    Construct the file of images to be ploted

    files: The path of the folder in which images are stored or list of paths of images or list of images
    m: Number of columns for the plot (if needed)
    n: Number of rows for the plot

    returns: Either list of paths of images or list of images
    """
    if seed:
        random.seed(seed)
    if isinstance(files, str):
        # Create a list of the filenames in the folder
        filenames = os.listdir(files)
        if m:
            k = n * m
        else:
            k = n
        sampled_filenames = random.sample(filenames, k)
    else:
        sampled_filenames = files
        if m:
            if not (n * m >= len(sampled_filenames)):
                warnings.warn(
                    "You won't be able to see all images due to lack of rows and columns"
                )
    return sampled_filenames


def load_image(
    files: str or List[str] or List[np.array],
    filename: str or np.array,
    source: str = "PNG",
) -> np.array:
    """
    Open the image of a given plot

    files: The path of the folder in which images are stored or list of paths of images or list of images
    filename: Either the path of the image or the image
    source: The source of image (PNG or DCOM)

    returns: np.array of image to plot at a given place
    """
    if source == "PNG":
        # If only the path is provided
        if isinstance(files, str):
            img = cv2.imread(os.path.join(files, filename))

        # If the list of images' path is provided
        elif isinstance(files, list) and all(isinstance(i, str) for i in files):
            img = cv2.imread(filename)

        # If the list of images is provided
        else:
            img = filename

    elif source == "DCOM":
        # TODO
        pass
    img = np.array(img)
    return img


def create_batch(
    df: pd.DataFrame, n: int = 64, label_ratio: float = 0.2, L_samples: List = None
):
    if not L_samples:
        n_positive = int(n * label_ratio)
        n_negative = n - n_positive
        df_sample_positive = df[df["cancer"] == 1].sample(n_positive)
        df_sample_negative = df[df["cancer"] == 0].sample(n_negative)
        df_sample = pd.concat([df_sample_positive, df_sample_negative])
        L_samples = df_sample.apply(
            lambda x: "".join(
                [img_path, "/", str(x["patient_id"]), "_", str(x["image_id"]), ".png"]
            ),
            axis=1,
        ).tolist()
    return [load_image(img_path, elem) for elem in L_samples]
