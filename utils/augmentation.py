import cv2
import numpy as np
from typing import List
import torch
from torchvision.transforms import *
import random

# From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9147240/
# Flipping
# Rotation
# Translation
# Scaling
# Erasing
# Kernel filter
# Advanced techniques
# a
# a
# a


class AugmentationPipeline(object):
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation: bool = True,
        translation: bool = True,
        scaling: bool = True,
        noise: bool = True,
        erasing: bool = False,
        kernel: bool = False,
        elastic_deformation: bool = False,
        random_apply: float = 0.7,
        #
        #
        #
        #
        # RandomCrop,
        # horizontal_flip: bool = True,
        # rotation: bool = True,
        **methods_args,
    ) -> None:
        """
        Constructor of the augmentation pipeline

        returns: None
        """

        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
        self.translation = translation
        self.scaling = scaling
        self.noise = noise
        self.erasing = erasing
        self.kernel = kernel
        self.elastic_deformation = elastic_deformation
        self.random_apply = random_apply
        self.vertical_flip = vertical_flip

        # Horizontal flip
        self.horizontal_flip_p = 0.5

        # Vertical flip
        self.horizontal_flip_p = 0.5

        # Rotation
        self.rotation_p = 0.5

        # Translation
        self.translation_p = 0.5
        self.translation_a = 0
        self.translation_b = 1 / 4
        assert self.translation_a <= self.translation_b

        # Scaling factor
        self.scaling_p = 0.5
        self.scale_a = 0.5
        self.scale_b = 2
        assert self.scale_a <= self.scale_b

        # Erasing
        self.erasing_scale = (0.02, 0.08)
        self.erasing_ratio = (0.3, 3.3)
        self.erasing_p = 0.1

        # Kernel
        self.kernel_method = "Mean filter"  # among ["Gausssian Noise","Mean filter", "Gaussian Blur", "Median Blur", "Laplacian Filter", "Poisson Noise"]
        if self.kernel_method == "Mean filter":
            self.kernel = torch.tensor(
                [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
            )

        # TODO other filters
        elif self.kernel_method == "Median filter":
            # TODO cf. https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
            pass

        # Possibilty to adjust parameters
        self.__dict__.update(methods_args)

    def __call__(self, sample: dict):
        """
        Preprocess function for Pytorch pipeline
        """

        # Horizontal flip
        if self.horizontal_flip:
            sample["image"] = RandomHorizontalFlip(self.horizontal_flip_p)(
                sample["image"]
            )

        # Vertical flip
        if self.vertical_flip:
            sample["image"] = RandomVerticalFlip(self.random_apply)(sample["image"])

        # Translation
        if self.translation and random.random() <= self.translation_p:
            sample["image"] = RandomAffine(
                degrees=(0, 0),
                translate=(self.translation_a, self.translation_b),
            )(sample["image"])

        # Scaling
        if self.scaling and random.random() <= self.scaling_p:
            sample["image"] = RandomAffine(
                degrees=(0, 0),
                scale=(self.scale_a, self.scale_b),
            )(sample["image"])

        # Rotation
        if self.rotation and random.random() <= self.rotation_p:
            sample["image"] = RandomRotation(degrees=(0, 180))(sample["image"])

        if self.erasing:
            sample["image"] = RandomErasing(
                p=self.erasing_p, scale=self.erasing_scale, ratio=self.erasing_ratio
            )(sample["image"])

        return sample


# pipe = AugmentationPipeline()
# arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# tens = torch.tensor(np.reshape(arr, (1, *arr.shape)))
# test = {"image": tens}
# print(pipe.__call__(test))

# Geometric Transformations ## TODO


# Pixel Level Augmentation ## TODO


# Kernel filters ## TODO


# Gan augmentation techiques ## TODO

#
