from torch import nn
import torch
from torch.optim.lr_scheduler import *

layers = [
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
]

# Cf pre_processing.py
pre_processing_parameters = {}

# Transform list
transform_list = []


# Scheduler
scheduler = ReduceLROnPlateau
scheduler_params = {"mode": "min", "factor": 0.1, "patience": 5}


def set_schduler(optimizer):
    return scheduler(optimizer, **scheduler_params)
