import sys
import os
import time
from time import sleep
import copy
import warnings
import random

sys.path.insert(0, "/appli")

import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim
from torch.nn import MSELoss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from utile.parser import parser
from utile.models import CustomEfficientNet
from utile.loaders import RNSADataset
from utile.pre_processing import PreProcessingPipeline
from torch.utils.data import DataLoader

from opencv_transforms import transforms

import wandb


# cf. https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/7 answer of Reuben Feinman
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True):
        assert len(y.shape) == 1, "label array must be 1D"
        n_batches = int(len(y) / batch_size)
        self.batch_size = batch_size
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            torch.randint(0, int(1e8), size=()).item()
        for _, indices in self.skf.split(self.X, self.y):
            yield indices

    def __len__(self):
        return len(self.y) // self.batch_size


class CustomBCELoss(torch.nn.Module):
    def __init__(self, weight_fn=None):
        super(CustomBCELoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        if weight_fn is None:
            weight_fn = lambda x: 1
        self.weight_fn = weight_fn

    def forward(self, input, target):
        weight = self.weight_fn(target)
        loss = self.loss_fn(input, target)
        weighted_loss = weight * loss
        return weighted_loss.mean()


def train_model(
    model: nn.Module,
    dataloaders: torch.utils.data.dataloader.DataLoader,
    criterion: torch.nn.modules.loss,
    optimizer: torch.optim,
    num_epochs: int = 15,
    include_features: bool = False,
    include_wandb: bool = False,
    device: str = "cpu",
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
    val_pf1_history = []
    if device == "cuda":
        device = torch.device("cuda")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_pf1 = 0.0
    for epoch in range(num_epochs):
        # print("Epoch {}/{}".format(epoch, num_epochs - 1))
        # print("-" * 10)

        # Each epoch has a training and validation phase

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_pf1 = 0.0
            running_corrects = 0
            phase = "train"
            # Iterate over data.
            with tqdm(enumerate(dataloaders[phase])) as tepoch:
                for ind, elem in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
                    # print(
                    #     f"Batch {ind+1} of {len(dataloaders[phase])} batches ", end="\r"
                    # )
                    if include_features:
                        inputs = {
                            k: elem[k].to(device)
                            for k in ["image", "features"]
                            if k in elem
                        }
                    else:
                        inputs = elem["image"]
                        inputs = inputs.to(device)
                    labels = elem["labels"]
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
                    running_corrects += torch.sum(preds == labels.data)
                    if include_features:
                        running_loss += loss.item() * inputs["image"].size(0)
                        running_pf1 += pfbeta(labels, outputs) * inputs["image"].size(0)
                    else:
                        running_loss += loss.item() * inputs.size(0)
                        running_pf1 += pfbeta(labels, outputs) * inputs.size(0)
                    try:
                        running_pf1 = running_pf1.item()
                        pf1 = pfbeta(labels, outputs).item()
                    except:
                        pf1 = pfbeta(labels, outputs)
                    # running_pf1 /= args.batch_size

                    # Display information of training
                    tepoch.set_postfix(
                        loss=loss.item(),
                        pf1=pf1,
                        batch=f"{ind+1}/{len(dataloaders[phase])}",
                    )
                    sleep(0.1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_pf1 = running_pf1 / len(dataloaders[phase].dataset)

            try:
                epoch_pf1 = epoch_pf1.item()
            except:
                pass

            print(
                f"Epoch {epoch} for {phase}: \t Loss: {epoch_loss:.4f}, pf1: {epoch_pf1:.4f}"
            )

            if phase == "val" and epoch_pf1 > best_pf1:
                if include_wandb:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": epoch_loss,
                        },
                        os.path.join(
                            wandb.run.dir, f"{args.model}_{wandb.run.name}.pt"
                        ),
                    )
                best_pf1 = epoch_pf1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_pf1_history.append(epoch_pf1)

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


if __name__ == "__main__":

    # Avoid useless warnings
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = torch.device("cuda")
    # Define the arguments' parser for training function
    args = parser.parse_args()
    if args.wandb:
        wandb.login()
        wandb.init(project="Kaggle_ RSNA", entity="turboteam")

    g = torch.Generator()
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        g.manual_seed(args.seed)
        # torch.use_deterministic_algorithms(True)

    # Define number of labels for prediction
    n_labels = len(args.labels)

    # CSV Training file
    train_df = pd.read_csv(args.csv_file_path)

    # Dataset with pre-processing pipeline and potential pytorch transforms
    transformed_dataset = RNSADataset(
        root_dir=args.images_dir,
        csv_file=args.csv_file_path,
        transform=transforms.Compose(
            [PreProcessingPipeline(**args.preprocessing_parameters), *args.transform]
        ),
    )

    # Load model
    if args.model == "EfficientNet":
        if args.layers:
            model = CustomEfficientNet(
                n_labels=n_labels,
                layers=args.layers,
                features=args.include_features,
                device=device,
            )
        else:
            model = CustomEfficientNet(
                n_labels=n_labels,
                features=args.include_features,
                device=device,
            )
    elif args.model == "ResNet":
        pass
        # TODO
    elif args.model == "ViT":
        pass
        # TODO
    # Define loss
    # TODO adapt loss if there are several labels
    if args.loss == "BCE":
        # weight_fn = lambda x: x * args.BCE_weights + (1 - x)
        weight_fn = lambda x: x + (1 - x) * args.BCE_weights
        loss = CustomBCELoss(weight_fn=weight_fn)
        # loss = BCELoss()
    elif args.loss == "MSE":
        loss = MSELoss()
    elif args.loss == "Custom":
        pass
        # TODO
    model.to(device)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.wandb:
        runs = wandb.Api().runs()
    try:
        num_runs = len(runs)
    except:
        num_runs = 0
    if args.wandb:
        wandb.run.name = f"Run {num_runs+1} with model {args.model}"
        wandb.watch(model, log_freq=100)
        wandb.config = args

    # If stratified sampling (to have the same ratio for classes between each batch)
    if args.stratified_sampling:

        # Creating data indices for training and validation splits:
        dataset_size = len(transformed_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(args.validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = np.array(indices[split:]), np.array(
            indices[:split]
        )

        # Set stratified samplers
        train_sampler = StratifiedBatchSampler(
            train_df["cancer"].iloc[train_indices],
            args.batch_size,
        )
        valid_sampler = StratifiedBatchSampler(
            train_df["cancer"].iloc[val_indices],
            args.batch_size,
        )

        # Define datasets
        final_dataset = {}
        final_dataset["train"] = Subset(transformed_dataset, train_indices)
        final_dataset["validation"] = Subset(transformed_dataset, val_indices)

        # Define dataloaders
        dataloader = {}
        dataloader["train"] = DataLoader(
            final_dataset["train"],
            num_workers=8,
            batch_sampler=train_sampler,
        )
        dataloader["validation"] = DataLoader(
            final_dataset["train"],
            num_workers=8,
            sampler=valid_sampler,
        )

    # If not stratified sampling
    else:
        # Define training and validation sets
        train_size = int((1 - args.validation_split) * len(transformed_dataset))
        validation_size = len(transformed_dataset) - train_size

        # Define dataset
        final_dataset = {}
        (
            final_dataset["train"],
            final_dataset["validation"],
        ) = torch.utils.data.random_split(
            transformed_dataset, [train_size, validation_size], generator=g
        )

        # Define dataloaders
        dataloader = {}
        dataloader["train"] = DataLoader(
            final_dataset["train"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
        )
        dataloader["validation"] = DataLoader(
            final_dataset["train"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
        )
    # if device == "cuda":
    #     model.to(torch.device("cuda"))
    train_model(
        model=model,
        dataloaders=dataloader,
        criterion=loss,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        include_features=args.include_features,
        include_wandb=args.wandb,
        device=device,
    )
