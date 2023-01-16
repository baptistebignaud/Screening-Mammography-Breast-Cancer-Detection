import argparse
from typing import List


parser = argparse.ArgumentParser(description="Training routine for models")

parser.add_argument(
    "--images_dir",
    type=str,
    help="Root directory of mammography screenings",
)

parser.add_argument(
    "--csv_file_path",
    type=str,
    help="Path for the training's CSV file",
)

parser.add_argument(
    "--model",
    type=str,
    choices=["EfficientNet", "ResNet", "ViT"],
    help="The type of mode you want to train/fine-tune",
)
parser.add_argument(
    "--layers",
    type=List,
    default=None,
    help="The list of classification layer that need to be taken into account",
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
    type=dict,
    default={},
    help="Parameters for the pre-processing pipeline",
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
    default=50,
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
    type=List,
    default=[],
    help="Additional transform that one could use with pytorch",
)

parser.add_argument(
    "--BCE_weights",
    type=float,
    default=3,
    help="How much false negatives should be penalized (false positives will be penalize with a factor 1)",
)

parser.add_argument(
    "--include_features",
    type=bool,
    default=True,
    help="If features need to be included as prior in the models",
)
