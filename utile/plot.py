import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import warnings


def plot_sample(
    files: str or List,
    m: int,
    n: int,
    fig_size: tuple = (15, 15),
    source: str = "PNG",
    wspace: float = 0.001,
    hspace: float = 0.001,
    seed: int = None,
) -> None:
    """
    Plot n*m randome sample of images or a list of images into a grid
    path: The path of the folder in which images are stored or list of paths to display
    m: Number of columns for the plot
    n: Number of rows for the plot
    fig_size: Figure's size
    source: Either PNG or DCOM format
    wspace: Width space between images in the figure
    hspace: Height space between images in the figure
    returns: None
    """
    if seed:
        random.seed(seed)
    if not (source in ["PNG", "DCOM"]):
        raise Exception("You need to choose a format among PNG or DCOM")

    if isinstance(files, str):
        # Create a list of the filenames in the folder
        filenames = os.listdir(files)
        k = n * m
        sampled_filenames = random.sample(filenames, k)
    else:
        sampled_filenames = files
        if not (n * m >= len(sampled_filenames)):
            warnings.warn(
                "You won't be able to see all images due to lack of rows and columns"
            )

    # Create the figure and axes objects that will hold the grid of images
    fig, axes = plt.subplots(n, m, figsize=fig_size)

    # Iterate through the sampled filenames and plot each image
    for i, filename in enumerate(sampled_filenames):
        # Load the image
        if source == "PNG":
            if isinstance(files, str):
                img = Image.open(os.path.join(files, filename))
            else:
                img = Image.open(filename)
        else:
            # TODO handle DCOM images
            pass
        # Plot the image on the appropriate subplot
        axes[i // m, i % m].imshow(img, cmap="gray")
        # Remove the axis labels
        axes[i // m, i % m].axis("off")
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.show()
