from typing import List
from torch import nn
import torch
from torch.nn import Conv2d
import warnings


class CustomEfficientNet(nn.Module):
    def __init__(
        self,
        n_labels: int,
        device: str = "cpu",
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
    ):
        """
        Efficientnet with custom layers

        n_labels: Number of labels used for the prediction (1 if only cancer but could be higher)
        layers: Layers to add for the classification part of the Efficientnet

        returns: None
        """
        super().__init__()
        self.features = features
        self.device = device

        l_linear = [layer for layer in layers if isinstance(layer, nn.Linear)]
        # if not (l_linear[0].weight.shape[1] == 1280):
        #     raise Exception(
        #         "Please provide custom classification layer with appropriate shape"
        #     )

        if l_linear[-1].weight.shape[0] < n_labels:
            warnings.warn(
                "The last provided layer has a lower dimension than the number of labels"
            )

        self.network = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            pretrained=True,
        )

        # Adjust to use Grayscale rather than RBG images
        self.network.stem.conv = Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        self.network.classifier.fc = nn.Linear(
            in_features=1280, out_features=1280, bias=True
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
