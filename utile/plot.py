import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import warnings
import numpy as np
import seaborn as sns
from scipy.stats import kde
import cv2
import math


def load_file(files: str or List[str] or List[np.array], n: int, m: int = None) -> List:
    """
    Construct the file of images to be ploted

    files: The path of the folder in which images are stored or list of paths of images or list of images
    m: Number of columns for the plot (if needed)
    n: Number of rows for the plot

    returns: Either list of paths of images or list of images
    """
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
            img = Image.open(os.path.join(files, filename))

        # If the list of images' path is provided
        elif isinstance(files, list) and all(isinstance(i, str) for i in files):
            img = Image.open(filename)

        # If the list of images is provided
        else:
            img = filename

    elif source == "DCOM":
        # TODO
        pass
    img = np.array(img)
    return img


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
    Plot n*m random sample of images or a list of images into a grid

    files: The path of the folder in which images are stored or list of paths of images or list of images
    n: Number of rows for the plot
    source: Either PNG or DCOM format
    title: The title of the whole figure
    wspace: Width space between images in the figure
    hspace: Height space between images in the figure
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
