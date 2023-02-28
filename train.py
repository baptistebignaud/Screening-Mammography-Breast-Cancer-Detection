import sys
import os
from os.path import join
import time
from time import sleep
import copy
import warnings
import random
from pathlib import Path

sys.path.insert(0, "./utils")

import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim
from torch.nn import MSELoss

from torch.utils.data import Subset

from utils.parser import parser
from utils.models import CustomModel
from utils.loaders import RNSADataset
from utils.pre_processing import PreProcessingPipeline
from torch.utils.data import DataLoader
from utils.augmentation import AugmentationPipeline
from utils.samplers import StratifiedBatchSampler, ImbalancedDatasetSampler
from utils.config import *
from utils.eval import pfbeta, CustomBCELoss
from opencv_transforms import transforms

import wandb
import pickle

import yaml

with open(r"private_config.yml") as file:
    yml_config_dict = yaml.load(file, Loader=yaml.FullLoader)


def train_model(
    model: nn.Module,
    dataloaders: torch.utils.data.dataloader.DataLoader,
    criterion: torch.nn.modules.loss,
    optimizer: torch.optim,
    num_epochs: int = 15,
    include_features: bool = False,
    include_wandb: bool = False,
    device: str = "cpu",
    scheduler_bool: bool = False,
    wandb_model=None,
    n_models: int = 1,
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
    # val_pf1_history = []
    if device == "cuda":
        device = torch.device("cuda")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_pf1 = 0.0
    if scheduler_bool:
        scheduler = set_schduler(optimizer)
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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_pf1 = running_pf1 / len(dataloaders[phase].dataset)
            if scheduler_bool:
                if scheduler == ReduceLROnPlateau:
                    scheduler.step(epoch_loss)
                else:
                    # TODO handle different types of scheduler
                    scheduler.step()

            try:
                epoch_pf1 = epoch_pf1.item()

            except:
                pass

            print(
                f"Epoch {epoch+1} for {phase}: \t Loss: {epoch_loss:.4f}, pf1: {epoch_pf1:.4f}"
            )

            if phase == "val":

                if epoch_pf1 >= best_pf1:
                    if include_wandb:
                        print("saving model")
                        torch.save(
                            {
                                "model": model,
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": epoch_loss,
                                "pf1": epoch_pf1,
                            },
                            # join(wandb.run.dir, f"{args.model}_{wandb.run.name}.pt"),
                            f"models/model_{n_models+1}_{wandb.run.name}.pt",
                        )

                        wandb.log_artifact(wandb_model)
                        wandb_model.wait()
                    best_pf1 = epoch_pf1
            if include_wandb:
                wandb.log(
                    {
                        f"{phase}_loss": epoch_loss,
                        f"{phase}_pf1": epoch_pf1,
                    }
                )

            # best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == "val":
            #     val_pf1_history.append(epoch_pf1)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val pf1: {:4f}".format(best_pf1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model  # , val_pf1_history


############################### Main ###############################


if __name__ == "__main__":
    n_models = None
    # Avoid useless warnings
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = torch.device("cuda")
    # Define the arguments' parser for training function
    args = parser.parse_args()
    if args.wandb:
        wandb.login()
        wandb.init(
            project=yml_config_dict["wandb_project"],
            entity=yml_config_dict["wandb_entity"],
        )

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

    # If pre processing and transform need to be applied
    if args.preprocessing_parameters:
        args.preprocessing_parameters = pre_processing_parameters
    else:
        args.preprocessing_parameters = {}

    if args.basic_augmentation:
        args.augmentation_parameters = pre_processing_parameters
    else:
        args.basic_augmentation = {}

    # Small adapatation for ViT
    if args.model == "ViT":
        args.preprocessing_parameters["resize"] = True
        args.preprocessing_parameters["resize_shape"] = ViTs[ViT_str]["resize_shape"]

    args.preprocessing_parameters["duplicate_channels"] = args.duplicate_channels

    if args.basic_augmentation:
        pre_processing_pipeline = PreProcessingPipeline(**args.preprocessing_parameters)
        augmentation_pipeline = AugmentationPipeline(**args.augmentation_parameters)

        transform = transforms.Compose(
            [
                pre_processing_pipeline,
                augmentation_pipeline,
            ]
        )

    else:
        pre_processing_pipeline = PreProcessingPipeline(**args.preprocessing_parameters)
        augmentation_pipeline = None
        transform = transforms.Compose(
            [PreProcessingPipeline(**args.preprocessing_parameters)]
        )

    # Dataset with pre-processing pipeline and potential pytorch transforms
    transformed_dataset = RNSADataset(
        root_dir=args.images_dir,
        csv_file=args.csv_file_path,
        transform=transform,
    )

    # Load model
    if args.layers:
        model = CustomModel(
            backbone=args.model,
            n_labels=n_labels,
            layers=layers,
            features=args.include_features,
            device=device,
            duplicate_channels=args.duplicate_channels,
            freeze_backbone=args.freeze_backbone,
        )
    else:
        model = CustomModel(
            backbone=args.model,
            n_labels=n_labels,
            features=args.include_features,
            device=device,
            duplicate_channels=args.duplicate_channels,
            freeze_backbone=args.freeze_backbone,
        )
    if args.wandb:
        s = "no " if not (args.stratified_sampling and args.multinomial_sampler) else ""
        f = "with" if args.include_features else "without"
        a = "with" if args.basic_augmentation else "without"
        wandb.run.name = f"Run with model {args.model} on {args.num_epochs} epochs and batch size of {args.batch_size} with {s} sampling {a} augmentation {f} features with penalization of false negatives of ratio {args.BCE_weights}"
        wandb.watch(model, log_freq=100)
        wandb.config = args

    if args.wandb:
        wandb_data = wandb.Artifact(
            name="RNSA_dataset",
            type="dataset",
            description="Information about the dataset with different preprocessing and transform parameters",
            metadata={
                "source": "https://www.kaggle.com/competitions/rsna-breast-cancer-detection",
                "pre_processing_parameters": pre_processing_pipeline.__dict__
                if isinstance(pre_processing_pipeline, PreProcessingPipeline)
                else None,
                "augmentation_parameters": augmentation_pipeline.__dict__
                if isinstance(augmentation_pipeline, AugmentationPipeline)
                else None,
            },
        )
        wandb_model = wandb.Artifact(
            name=f"{args.model}",
            type="model",
            description=f"Information about the model with backbone {args.model} used for training",
            metadata={
                "Parser configuration": vars(args),
                "Classification Layers": layers,
                "Backbone": EfficientNet_str
                if args.model == "EfficientNet"
                else ViT_str
                if args.model == "ViT"
                else ResNet_str,
            },
        )
        n_models = len([name for name in os.listdir("models")])
        torch.save(
            {
                "model": model,
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": 0,
                "loss": 0,
                "pf1": 0,
            },
            # join(wandb.run.dir, f"{args.model}_{wandb.run.name}.pt"),
            f"models/model_{n_models+1}_{wandb.run.name}.pt",
        )
        n_encoders = len([name for name in os.listdir("ohe_encoders")])
        with open(f"ohe_encoders/encoder_{n_encoders}.pkl", "wb") as fp:
            pickle.dump(transformed_dataset.get_encoders(), fp)
        wandb_data.add_file(Path(f"ohe_encoders/encoder_{n_encoders}.pkl"))
        wandb.log_artifact(wandb_data)
    else:
        wandb_model = None
    # Avoid having weights with multinomial sampler
    if args.multinomial_sampler:
        args.BCE_weights = args.multinomial_sampler_BCE_weights
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

    # If stratified sampling (to have the same ratio for classes between each batch)
    if args.stratified_sampling or args.multinomial_sampler:

        # Creating data indices for training and validation splits:
        dataset_size = len(transformed_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(args.validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = np.array(indices[split:]), np.array(
            indices[:split]
        )

        # Define datasets
        final_dataset = {}
        final_dataset["train"] = Subset(transformed_dataset, train_indices)
        final_dataset["val"] = Subset(transformed_dataset, val_indices)

        if args.multinomial_sampler:

            train_sampler = ImbalancedDatasetSampler(
                labels=train_df["cancer"].iloc[train_indices],
                batch_size=args.batch_size,
            )
            valid_sampler = ImbalancedDatasetSampler(
                labels=train_df["cancer"].iloc[val_indices],
                batch_size=args.batch_size,
            )
        else:
            # Set stratified samplers
            train_sampler = StratifiedBatchSampler(
                train_df["cancer"].iloc[train_indices],
                args.batch_size,
            )
            valid_sampler = StratifiedBatchSampler(
                train_df["cancer"].iloc[val_indices],
                args.batch_size,
            )

        # Define dataloaders
        dataloader = {}
        dataloader["train"] = DataLoader(
            final_dataset["train"],
            num_workers=8,
            batch_sampler=train_sampler,
        )
        dataloader["val"] = DataLoader(
            final_dataset["val"],
            num_workers=8,
            batch_sampler=valid_sampler,
        )

    # If not stratified sampling
    else:
        # Define training and validation sets
        train_size = int((1 - args.validation_split) * len(transformed_dataset))
        validation_size = len(transformed_dataset) - train_size

        # Define dataset
        final_dataset = {}
        (final_dataset["train"], final_dataset["val"],) = torch.utils.data.random_split(
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
        dataloader["val"] = DataLoader(
            final_dataset["train"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
        )
    # if device == "cuda":
    #     model.to(torch.device("cuda"))
    if args.wandb:
        wandb_model.add_file(Path(f"models/model_{n_models+1}_{wandb.run.name}.pt"))
    train_model(
        model=model,
        dataloaders=dataloader,
        criterion=loss,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        include_features=args.include_features,
        include_wandb=args.wandb,
        device=device,
        scheduler_bool=args.lr_scheduler,
        wandb_model=wandb_model,
        n_models=n_models,
    )
