from typing import List
from torch import nn
import torch
from torch.nn import Conv2d
import warnings
from config import ViT_str, ViTs, ResNet_str, EfficientNet_str
import cv2
import numpy as np
from opencv_transforms import transforms

list_available_backbone = ["EfficientNet", "ViT", "ResNet"]


class CustomModel(nn.Module):
    def __init__(
        self,
        n_labels: int,
        device: str = "cpu",
        backbone: str = "EfficientNet",
        features: bool = False,
        layers: List = [
            nn.Linear(in_features=1024, out_features=512, bias=True),
            torch.nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            torch.nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            torch.nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            torch.nn.Dropout(p=0.3),
            nn.LeakyReLU(),
        ],
        nb_channels: int = 1,
        pretrained: bool = True,
    ):
        """
        Custom models with custom layers

        n_labels: Number of labels used for the prediction (1 if only cancer but could be higher)
        layers: Layers to add for the classification part of the Efficientnet
        device: Device on which to train model
        backbone: Backbone model
        features: If to include prior features for the training or not (like age...)
        layers: Classification layers on top of backbone model
        nb_channels: Number of channels of the input image (could be useful if one wants to add channels to the image)
        pretrained: If load pre_trained model or not

        returns: None
        """
        super().__init__()
        if not backbone in list_available_backbone:
            raise Exception(
                f"Please provide a backbone model in {list_available_backbone}"
            )
        self.backbone = backbone
        self.features = features
        self.device = device
        self.n_labels = n_labels
        self.nb_channels = nb_channels
        self.pretrained = pretrained

        l_linear = [layer for layer in layers if isinstance(layer, nn.Linear)]
        # if not (l_linear[0].weight.shape[1] == 1280):
        #     raise Exception(
        #         "Please provide custom classification layer with appropriate shape"
        #     )

        if l_linear[-1].weight.shape[0] < n_labels:
            warnings.warn(
                "The last provided layer has a lower dimension than the number of labels"
            )
        if self.backbone == "EfficientNet":
            self.network = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub",
                EfficientNet_str,
                pretrained=self.pretrained,
            )

            # Adjust to use Grayscale rather than RBG images
            self.network.stem.conv = Conv2d(
                self.nb_channels,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )

            self.network.classifier.fc = nn.Linear(
                in_features=1280, out_features=1280, bias=True
            )
        elif self.backbone == "ViT":
            vit_weights = None
            if self.pretrained:
                vit_weights = ViTs[ViT_str]["weights"]
            self.network = ViTs[ViT_str]["model"](weights=vit_weights)

            self.network.conv_proj = Conv2d(
                self.nb_channels,
                self.network.hidden_dim // ((self.network.patch_size) ** 2),
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # self.network.heads = nn.Linear(
            #     in_features=1280, out_features=1280, bias=True
            # )
        elif self.backbone == "ResNet":
            self.network = torch.hub.load(
                "pytorch/vision:v0.10.0", ResNet_str, pretrained=self.pretrained
            )
            self.network.conv1 = Conv2d(
                self.nb_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        # Replace last layer
        self.classifier = nn.Sequential(
            *layers,
            nn.Linear(l_linear[-1].weight.shape[0], n_labels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.features:
            im = self.network(x["image"])
            out = torch.cat((im, x["features"]), dim=1).to(self.device)
        else:
            out = self.network(x)
        out = nn.Linear(out.shape[1], self.classifier[0].weight.shape[1]).to(
            self.device
        )(out)
        out = self.classifier(out)
        return out
