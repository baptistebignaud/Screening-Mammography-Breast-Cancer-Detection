from typing import List
from torch import nn
import torch
from torch.nn import Conv2d
import time
import copy
import warnings


class CustomEfficientNet(nn.Module):
    def __init__(
        self,
        n_labels: int,
        layers: List = [
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
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

        l_linear = [layer for layer in layers if isinstance(layer, nn.Linear)]
        if not (l_linear[0].weight.shape[1] == 1280):
            raise Exception("Please custom classification layer with appropriate shape")

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

        # Replace last layer
        self.network.classifier.fc = nn.Sequential(
            *layers,
            nn.Linear(l_linear[-1].weight.shape[0], n_labels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.network(x)
        return out


def train_model(
    model: nn.Module,
    dataloaders: torch.utils.data.dataloader.DataLoader,
    criterion: torch.nn.modules.loss,
    optimizer: torch.optim,
    num_epochs: int = 15,
) -> tuple:
    """
    Function to train models and keeps track of training

    dataloaders: Dataloader of the dataset
    criterion: Loss function to use
    optimizer: The optimizer used to train the model
    num_epochs: Number of epochs

    returns: The best model with history of val pf1
    """
    since = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_pf1_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_pf1 = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase

        # for phase in ["train", "val"]:
        #     if phase == "train":
        #         model.train()  # Set model to training mode
        #     else:
        #         model.eval()  # Set model to evaluate mode

        #     running_loss = 0.0
        #     running_corrects = 0

        running_loss = 0.0
        running_pf1 = 0.0
        running_corrects = 0
        phase = "train"
        # Iterate over data.
        for ind, elem in enumerate(dataloaders):
            print(f"Batch {ind+1} of {len(dataloaders)} batches ", end="\r")
            inputs = elem["image"]
            labels = elem["labels"]

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_pf1 += pfbeta(labels, outputs) * inputs.size(0)

        # epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_loss = running_loss / len(dataloaders.dataset)

        # epoch_pf1 = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_pf1 = running_pf1 / len(dataloaders.dataset)
        try:
            epoch_pf1 = epoch_pf1.item()
        except:
            pass

        print("{} Loss: {:.4f} pf1: {:.4f}".format(phase, epoch_loss, epoch_pf1))

        # deep copy the model
        if phase == "val" and epoch_pf1 > best_pf1:
            best_pf1 = epoch_pf1
            best_model_wts = copy.deepcopy(model.state_dict())
        if phase == "val":
            val_pf1_history.append(epoch_pf1)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val pf1: {:4f}".format(best_pf1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_pf1_history


def pfbeta(labels, predictions, beta: float = 1):
    """
    Official implementation of the evaluation metrics, pf1 Score,
    cf. https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview/evaluation
    """
    y_true_count = 0
    ctp = 0
    cfp = 0
    for idx in range(len(labels)):

        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    # Add if ever there is no true prediction to avoid divide by 0
    if y_true_count == 0:
        return 0

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0
