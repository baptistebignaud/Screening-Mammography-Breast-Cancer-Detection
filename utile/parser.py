import argparse
from typing import List


parser = argparse.ArgumentParser(description="Training routine for models")

parser.add_argument(
    "--images_dir",
    type=str,
    required=True,
    help="Root directory of mammography screenings",
)

parser.add_argument(
    "--csv_file_path",
    type=str,
    required=True,
    help="Path for the training's CSV file",
)

parser.add_argument(
    "--model",
    type=str,
    default="EfficientNet",
    choices=["EfficientNet", "ResNet", "ViT"],
    help="The type of mode you want to train/fine-tune",
)
parser.add_argument(
    "--layers",
    action="store_true",
    help="If to change default layers in classifier for models",
)

parser.add_argument(
    "--loss",
    type=str,
    default="BCE",
    choices=["BCE", "MSE", "Custom"],
    help="Loss function to train the model",
)

parser.add_argument(
    "--labels",
    type=List,
    default=["cancer"],
    help="Number of labels that need to be predicted (1 if one wants to only predit label cancer, could be more)",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="Learning rate of the optimizer",
)

parser.add_argument(
    "--preprocessing_parameters",
    action="store_true",
    help="If custom parameters for the pre-processing pipeline need to be used",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size for training",
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="Number of epochs for training",
)

parser.add_argument(
    "--validation_split",
    type=float,
    default=0.1,
    help="Ratio that would be taken into account for validation split during training",
)

parser.add_argument(
    "--seed",
    type=int,
    default=53,
    help="Seed for reproductible results",
)

parser.add_argument(
    "--transform",
    action="store_true",
    help="Additional transform that one could use with pytorch",
)

parser.add_argument(
    "--BCE_weights",
    type=float,
    default=50,
    help="How much false negatives should be penalized (false positives will be penalize with a factor 1)",
)

parser.add_argument(
    "--include_features",
    action="store_true",
    help="If features need to be included as prior in the models",
)

parser.add_argument(
    "--wandb",
    action="store_true",
    help="If wandb need to be set up to save models",
)

parser.add_argument(
    "--stratified_sampling",
    action="store_true",
    help="If stratified sampling needs to be used for data generator",
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Number of available workers for dataloader",
)

parser.add_argument(
    "--lr_scheduler",
    action="store_true",
    help="If a scheduler needs to be used for the training optimzer",
)

parser.add_argument(
    "--duplicate_channels",
    action="store_true",
    help="Either to use rgb or duplicate channels",
)
