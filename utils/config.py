from torch import nn
import torch
from torch.optim.lr_scheduler import *
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    ViT_H_14_Weights,
)

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

# Cf augmentation.py
augmentation_parameters = {}


################## backbone models ##################
######### EfficientNet #########
EfficientNet_str = "nvidia_efficientnet_b0"
# among [nvidia_efficientnet_b0,nvidia_efficientnet_b4,nvidia_efficientnet_widese_b0,nvidia_efficientnet_widese_b4]
# cf. https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/

######### ViT #########
# Value to change
ViT_str = "vit_b_16"
# h14 not working because need to readjust hidden dimension to be divisible by patch_size (or to readjust patch_size)
# See line 279 in https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py for more information
# Among ["vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"]
# for more information cf. https://pytorch.org/vision/main/models/vision_transformer.html

ViTs = {
    "vit_h_14": {
        "model": vit_h_14,
        "weights": ViT_H_14_Weights.DEFAULT,
        "resize_shape": 518,
    },
    "vit_b_16": {
        "model": vit_b_16,
        "weights": ViT_B_16_Weights.DEFAULT,
        "resize_shape": 224,
    },
    "vit_b_32": {
        "model": vit_b_32,
        "weights": ViT_B_32_Weights.DEFAULT,
        "resize_shape": 224,
    },
    "vit_l_16": {
        "model": vit_l_16,
        "weights": ViT_L_16_Weights.DEFAULT,
        "resize_shape": 224,
    },
    "vit_l_32": {
        "model": vit_l_32,
        "weights": ViT_L_32_Weights.DEFAULT,
        "resize_shape": 224,
    },
}

######### Resnet #########
ResNet_str = "resnet18"
# Among [resnet18,resnet34,resnet50,resnet101,resnet152] cf. https://pytorch.org/hub/pytorch_vision_resnet/


# Scheduler
scheduler = ReduceLROnPlateau
scheduler_params = {"mode": "min", "factor": 0.1, "patience": 5}


def set_schduler(optimizer):
    return scheduler(optimizer, **scheduler_params)


cat_features = ["machine_id", "laterality", "view"]
