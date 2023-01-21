from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from typing import List
import numpy as np
import warnings
from PIL import Image
import os
import cv2
import pandas as pd
import torch
from pre_processing import PreProcessingPipeline
from opencv_transforms import transforms


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


class RNSADataset(Dataset):
    """
    RNSA dataset
    """

    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        features=["age", "implant", "machine_id", "laterality", "view"],
        labels=["cancer"]
        # , "biopsy", "invasive", "BIRADS", "difficult_negative_case"],
    ):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csv_file)
        df["path"] = df.apply(
            lambda x: "".join(
                [root_dir, "/", str(x["patient_id"]), "_", str(x["image_id"]), ".png"]
            ),
            axis=1,
        )
        self.ohe_columns = pd.get_dummies(
            df[list(set(features) & set(["machine_id", "laterality", "view"]))],
            columns=list(set(features) & set(["machine_id", "laterality", "view"])),
        ).columns
        df = pd.get_dummies(
            df,
            columns=list(set(features) & set(["machine_id", "laterality", "view"])),
        )
        self.RNSA_frame = df
        self.root_dir = root_dir
        self.transform = transform
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.RNSA_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.RNSA_frame.iloc[idx]["path"])
        image = cv2.imread(img_name)
        features = self.RNSA_frame.iloc[idx][self.ohe_columns].to_numpy()

        # features = features

        labels = self.RNSA_frame.iloc[idx][self.labels].to_numpy()
        # features = features.reshape(-1, 2)
        # labels = labels.reshape(-1, 2)
        sample = {"image": image, "features": features, "labels": labels}

        if self.transform:
            sample = self.transform(sample)
        return sample
