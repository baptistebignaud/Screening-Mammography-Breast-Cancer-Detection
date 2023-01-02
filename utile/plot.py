# os and diverse librairies
import os
from os import listdir
import warnings
import random
from typing import List
from random import sample

# Images librairies
import cv2

# Plot librairies
import seaborn as sns
import matplotlib.pyplot as plt

# Math and data manipulation librairies
import numpy as np
import math
import pandas as pd

# Librairies for basics manipulation on file
import six
from utile.loaders import load_file, load_image


def plot_sample(
    files: str or List[str] or List[np.array],
    m: int,
    n: int,
    fig_size: tuple = (15, 15),
    source: str = "PNG",
    title: str = None,
    wspace: float = 0.04,
    hspace: float = 0.04,
    seed: int = None,
    font_size: int = 20,
    normalize: bool = False,
) -> None:
    """
    Plot n*m random sample of images or a list of images into a grid

    files: The path of the folder in which images are stored or list of paths of images or list of images
    m: Number of columns for the plot
    n: Number of rows for the plot
    fig_size: Figure's size
    source: Either PNG or DCOM format
    title: The title of the whole figure
    wspace: Width space between images in the figure
    hspace: Height space between images in the figure
    seed: Seed of the randomness if one wants reproductible results
    normalize: If one wants to normalize the image
    font_size: Font's size of title of the figure

    returns: None
    """

    if seed:
        random.seed(seed)
    if not (source in ["PNG", "DCOM"]):
        raise Exception("You need to choose a format among PNG or DCOM")

    # Get the names of images to plot or directly images
    sampled_filenames = load_file(files, m=m, n=n)

    # Create the figure and axes objects that will hold the grid of images
    fig, axes = plt.subplots(n, m, figsize=fig_size)

    if title:
        # set the global title, font size, and font name
        fig.suptitle(title, fontsize=font_size)

    # Iterate through the sampled filenames and plot each image
    for i, filename in enumerate(sampled_filenames):
        # Load the image
        img = load_image(files, filename)
        # Normalize if needed
        if normalize:
            try:
                # If image is in color
                img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            except:
                # If image is already in grayscale
                img = cv2.equalizeHist(img)
        # Plot the image on the appropriate subplot
        axes[i // m, i % m].imshow(img, cmap="gray")
        # Remove the axis labels
        axes[i // m, i % m].axis("off")
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.show()


def plot_img_hist(
    files: str or List[str] or List[np.array],
    n: int,
    widths: List = [20, 20],
    heights: List = [20, 10],
    source: str = "PNG",
    seed: int = None,
    font_size: int = 12,
    normalize: bool = False,
    nb_bins_hist: int = 20,
) -> None:
    """
    Plot n plots of images and associated histograms and kde plot

    files: The path of the folder in which images are stored or list of paths of images or list of images
    n: Number of rows for the plot
    widths: List of widths for figures
    heights: List of heights for figures
    source: Either PNG or DCOM format
    seed: Seed of the randomness if one wants reproductible results
    font_size: Font's size of title of the figure
    normalize: If one wants to normalize the image
    nb_bins_hist: Number of bins for the histogram

    returns: None
    """
    sns.set_style("dark")
    if seed:
        random.seed(seed)
    if not (source in ["PNG", "DCOM"]):
        raise Exception("You need to choose a format among PNG or DCOM")

    # Get the names of images to plot or directly images
    sampled_filenames = load_file(files, n=n)

    # Iterate through the sampled filenames and plot each image
    for i, filename in enumerate(sampled_filenames):
        # Create the figure and axes objects that will hold the grid of images
        fig = plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(
            ncols=2, nrows=2, width_ratios=widths, height_ratios=heights
        )
        # Load the image
        img = load_image(files, filename)
        # Normalize if needed
        if normalize:
            try:
                img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            except:
                img = cv2.equalizeHist(img)

        # Plot of the raw image
        ax = fig.add_subplot(spec[0, :])
        # Title includes id if we have it
        if isinstance(files, str) or (
            isinstance(files, list) and all(isinstance(i, str) for i in files)
        ):
            title = f"Mammography screening of image {sampled_filenames[i]}"
        else:
            title = "Mammography screening"
        ax.set_title(title)
        ax.title.set_size(font_size)
        plt.imshow(img, cmap="gray")

        # Histogram
        ax = fig.add_subplot(spec[1, 0])
        sns.histplot(
            img[np.where(img != 0)].flatten(), bins=nb_bins_hist, stat="percent"
        )
        ax.set_xlabel("Value of pixels")
        ax.set_title("Histogram of image")

        # Kde plot
        ax = fig.add_subplot(spec[1, 1])
        sns.kdeplot(data=img[np.where(img != 0)].flatten(), multiple="stack")
        ax.set_title("Image kernel density plot")
        ax.set_xlabel("Value of pixels")

        plt.show()


def plot_hist_3D(
    files: str or List[str] or List[np.array],
    n: int,
    source: str = "PNG",
    seed: int = None,
    smoothen: bool = False,
    nbins: int = 100,
) -> None:
    """
    Plot a 3D views of stacked histograms of n images, this is completely experimental, it is to try to
    have a more general view of histograms and how they are different. Be aware that smoothen part is
    also very exprimental and has high complexity (it closen images that are more similar)

    files: The path of the folder in which images are stored or list of paths of images or list of images
    n: Number of rows for the plot
    source: Either PNG or DCOM format
    seed: Seed of the randomness if one wants reproductible results
    normalize: If one wants to normalize the image
    smoothen: Boolean to tell if the curve must be smoothened or not
    nb_bins: Number of bins for histograms

    returns: None
    """
    sys.modules["sklearn.externals.six"] = six
    import mlrose
    from scipy.interpolate import griddata
    import plotly.offline as pyo
    import plotly.io as pio
    import plotly.graph_objects as go

    if seed:
        random.seed(seed)
    if not (source in ["PNG", "DCOM"]):
        raise Exception("You need to choose a format among PNG or DCOM")
    if smoothen and n >= 30:
        warnings.warn(
            "The number of images is high to smooth, and it might take too much time"
        )
    # Get the names of images to plot or directly images
    sampled_filenames = load_file(files, n=n)
    imgs = [load_image(files=files, filename=file) for file in sampled_filenames]

    histograms = np.array(
        [np.histogram(img[np.where(img > 30)].flatten(), bins=nbins)[0] for img in imgs]
    )
    coordinates = np.array(
        [np.histogram(img[np.where(img > 3)].flatten(), bins=nbins)[1] for img in imgs]
    )
    # Rerange coordinates of histograms to be in the midle of the bar
    coordinates2 = np.array(
        [
            [
                (coordinates[i, j] + coordinates[i, j + 1]) / 2
                for j in range(coordinates.shape[1] - 1)
            ]
            for i in range(coordinates.shape[0])
        ]
    )
    if smoothen:
        # Creation of histograms of images to see the distance between images repartition of pixels (with MSE)
        L = np.array([np.histogram(img.flatten(), bins=nbins)[0] for img in imgs])
        Dist = np.array(
            [
                [math.sqrt(np.square(np.subtract(elem1, elem2)).mean()) for elem1 in L]
                for elem2 in L
            ]
        )

        # Optimization model doesn't accept 0 on diagonal terms
        for i in range(Dist.shape[0]):
            Dist[i][i] = 10e8

        # Reorganization of format of distances so that it could be trained
        Dist_mlrose = [
            (i, j, Dist[i, j])
            for i in range(Dist.shape[0])
            for j in range(Dist.shape[1])
        ]

        fitness_dists = mlrose.TravellingSales(distances=Dist_mlrose)
        problem_fit = mlrose.TSPOpt(length=n, fitness_fn=fitness_dists, maximize=False)

        # Get the best ordering (here genetic algorithm)
        best_state, best_fitness = mlrose.genetic_alg(
            problem_fit, mutation_prob=0.2, max_attempts=30, random_state=2
        )

        # Rerange order with smoothing
        histograms = histograms[best_state]
        coordinates = coordinates[best_state]
    # Space the histograms in the 3D space
    width = 1
    coordinateY = np.array(
        [
            [width * n for i in range(coordinates2.shape[1])]
            for n in range(coordinates.shape[0])
        ]
    )
    # Dimension of coordinates of histograms
    x = coordinates2.flatten()
    # Dimension of the images (each unit*width is a histogram from one image)
    y = coordinateY.flatten()
    # Number of pixels in the histogram
    z = histograms.flatten()

    # Meshgrid for the 3D surface
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # Gridata with interpolation
    Z = griddata((x, y), z, (X, Y), method="cubic")

    # Plot the 3D surface
    pio.renderers.default = "iframe"
    fig = go.Figure(go.Surface(x=xi, y=yi, z=Z))
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            yaxis=dict(
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            zaxis=dict(
                backgroundcolor="rgb(230, 230,200)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
        ),
        width=700,
        margin=dict(r=10, l=10, b=10, t=10),
    )

    # Add axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title="Coordinates in the histograms",
            yaxis_title="Sequence of images/histograms",
            zaxis_title="Number of pixels",
        ),
        font=dict(color="black", size=10),
    )

    # Add title
    fig.update_layout(
        title=dict(
            text="3D representation of histograms of n images",
            font=dict(color="black", size=20),
            xanchor="center",
            yanchor="top",
            y=0.9,
            x=0.5,
        )
    )

    fig.show()


def build_list_imgs(
    df: pd.DataFrame,
) -> tuple:
    """
    Construct list of dataframes of images (filtered by laterality, view and
    if cancer or not) with associated name

    df: Dataframe with metadata about images

    returns: (List of filtered dataframes images, associated names for labels )
    """

    # Get possible values and associated labels for each filter
    cancer, cancer_labels = [0, 1], ["without cancer", "with cancer"]
    laterality, laterality_labels = ["L", "R"], ["Left view", "Right view"]
    view, view_labels = ["MLO", "CC"], [
        "Mediolateral Oblique",
        "Cranial Caudal",
    ]

    # Get list of dataframes with filters
    L_imgs = [
        df[(df["laterality"] == lat) & ((df["view"] == v)) & ((df["cancer"] == c))]
        for c in cancer
        for v in view
        for lat in laterality
    ]

    # Get associated names
    L_names = [
        f"{lat} {v} {c}"
        for c in cancer_labels
        for v in view_labels
        for lat in laterality_labels
    ]
    return L_imgs, L_names


def plot_avg_imgs(
    L_imgs: List,
    L_names: List,
    n_samples: int = 100,
    widths: List[int] = [10, 10],
    heights: List[int] = [40, 40, 40, 40],
    title_size: int = 10,
    fig_size: tuple = (15, 15),
    seed: int = None,
) -> None:
    """
    Plot average images for each filter onto a grid to see for each
    laterality, each view and if has cancer or not, the average
    image.

    L_imgs: List of filtered dataframe (output of build_list_imgs)
    L_names: List of associated names (output of build_list_imgs)
    n_samples: Number of samples for the average image
    widths: List of widths for figures
    heights: List of heights for figures
    title_size: Size of the main title
    fig_size: Figure size
    seed: Seed of the randomness if one wants reproductible results

    returns: List of images by
    """
    if seed:
        random.seed(seed)
    if not (len(widths) == 2 and len(heights) == 4):
        raise Exception("List of widths or heights ratios has not the correct length")

    # Get a sample for each filtered dataframe
    # TODO check that the number of samples is lower that the shape
    L_samples = [elem.sample(n=n_samples) for elem in L_imgs]

    # Create list of average image after histogram normalization
    mean_images = []
    for sample_elem in L_samples:
        list_for_mean = []
        for file in sample_elem["path"]:
            img = cv2.imread(file)
            img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            list_for_mean.append(img)
        list_for_mean = np.array(list_for_mean)
        mean_img = np.mean(list_for_mean, axis=0) / 255
        mean_images.append(mean_img)
    mean_images = np.array(mean_images)

    sns.set_style("dark")

    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=fig_size)

    spec = fig.add_gridspec(
        ncols=2, nrows=4, width_ratios=widths, height_ratios=heights
    )

    # Parameters for the titles
    params_col = ["Without cancer", "With cancer"]
    params_rows = [
        "Left mediolateral Oblique",
        "Right mediolateral Oblique",
        "Left Cranial Caudal",
        "Right Cranial Caudal",
    ]
    for row in range(0, 4):
        for col in range(0, 2):

            ax = fig.add_subplot(spec[row, col])

            # Add columns' titles
            if row == 0:
                ax.annotate(
                    params_col[col],
                    xy=(0.5, 1),
                    xytext=(0, 40),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                    fontsize=30,
                    weight="bold",
                )

            # Add rows' titles
            if col == 0:
                ax.annotate(
                    params_rows[row],
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 120, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                    fontsize=15,
                    weight="bold",
                )
            # Set for each plot the title
            ax.set_title(
                L_names[
                    (col + 2 * row) // 2 + ((len(L_names) // 2) * ((col + 2 * row) % 2))
                ]
            )
            ax.title.set_size(title_size)
            # Plot image
            plt.imshow(
                mean_images[
                    (col + 2 * row) // 2 + ((len(L_names) // 2) * ((col + 2 * row) % 2))
                ],
                cmap="gray",
            )
