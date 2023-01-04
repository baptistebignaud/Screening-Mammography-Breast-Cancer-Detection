### file to generate dataloaders
### for simplicity, we do not offer the choice to load the whole dataset in VRAM anymore

from torchvision import transforms, datasets
import random
from args import args
import torch
import torch.nn as nn
import pandas as pd
import os
import json
import numpy as np
from PIL import Image
import copy
from selfsupervised.selfsupervised import get_ssl_transform
from utils import *
from few_shot_evaluation import EpisodicGenerator
from augmentations import parse_transforms
### first define dataholder, which will be used as an argument to dataloaders
all_steps = [item for sublist in eval(args.steps) for item in sublist]
supervised = 'lr' in all_steps or 'rotations' in all_steps or 'mixup' in all_steps or 'manifold mixup' in all_steps or (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "" or args.episodic
class DataHolder():
    def __init__(self, data, targets, transforms, target_transforms=lambda x:x, opener=lambda x: Image.open(x).convert('RGB')):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert(self.length == len(targets))
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.opener = opener
    def __getitem__(self, idx):
        if isinstance(self.data[idx], str):
            elt = self.opener(args.dataset_path + self.data[idx])
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.target_transforms(self.targets[idx])
    def __len__(self):
        return self.length

def dataLoader(dataholder, shuffle, datasetName):
    return torch.utils.data.DataLoader(dataholder, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(os.cpu_count(), 8))

class TransformWrapper(object):
    """
    Wrapper for different transforms.
    """
    def __init__(self, all_transforms):
        self.all_transforms = all_transforms
    def __call__(self, image):
        out = {}
        for name, T in self.all_transforms.items():
            out[name] = T(image)
        return out

def get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms):
    if datasetName == 'train':
        supervised_transform_str = args.training_transforms if len(args.training_transforms) > 0 else default_train_transforms
        supervised_transform = parse_transforms(supervised_transform_str, image_size) 
        all_transforms = {}
        if supervised:
            all_transforms['supervised'] = transforms.Compose(supervised_transform)
        all_transforms.update(get_ssl_transform(image_size, normalization=supervised_transform[-1]))
        trans = TransformWrapper(all_transforms)
    else:
        trans = transforms.Compose(parse_transforms(args.test_transforms if len(args.test_transforms) > 0 else default_test_transforms, image_size))
    return trans

def imagenet(datasetName):
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 224
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 224
    default_train_transforms = ['randomresizedcrop','randomhorizontalflip', 'totensor', 'imagenetnorm']
    if args.sample_aug == 1:
        default_test_transforms = ['resize_256/224', 'centercrop', 'totensor', 'imagenetnorm']
    else:
        default_test_transforms = ['randomresizedcrop', 'totensor', 'imagenetnorm'] 
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)
    pytorchDataset = datasets.ImageNet(args.dataset_path + "/imagenet", split = "train" if datasetName != "test" else "val", transform = trans)
        
    return {"dataloader": dataLoader(pytorchDataset, shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="imagenet_"+datasetName), "name":"imagenet_" + datasetName, "num_classes":1000, "name_classes": pytorchDataset.classes}

def rsna(rsna_name, split, is_train):
    f = open(args.dataset_path + "datasets_rsna.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets[rsna_name]
    datasetName=f"{rsna_name}_"+("train" if is_train else "validation")
    if is_train:
        image_size = args.training_image_size if args.training_image_size>0 else 224
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 224
    default_train_transforms = ['resize','randomhorizontalflip', 'colorjitter', 'randomverticalflip', 'totensor']
    if args.sample_aug == 1:
        default_test_transforms = ['resize', 'totensor']
    else:
        default_test_transforms = ['randomresizedcrop', 'totensor']

    # depeding the split, we need to select the right data
    start, end = split
    len_data = len(dataset["data"])
    start = int(start/100 * len_data)
    end = int(end/100 * len_data)
    data = dataset["data"][start:end]
    targets = dataset["targets"][start:end]

    trans = get_transforms(image_size, 'train' if is_train else 'validation', default_train_transforms, default_test_transforms)
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = is_train, datasetName=datasetName), "name":datasetName, "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def prepareDataLoader(name, is_train=False):
    if isinstance(name, str):
        name = [name]
    result = []
    train_trans_results = []
    dataset_options = {
            "imagenet_train": lambda: imagenet("train"),
            "imagenet_validation": lambda: imagenet("validation"),
            "imagenet_test": lambda: imagenet("test"),
        }
    for elt in name:
        if 'rsna' in elt: # since rsna has only a training set, we need to split it using a k fold cross validation depending on the iput argument which can be rsna_start_val_end_val
            if '_' in elt:
                split = elt.split('_')[1:3]
                split = [int(elt) for elt in split]
            else:
                split = [0, 100] # default split is all training data and no validation data   
            assert split[0]<=split[1], f'Please provide a valid split for rsna, the first value should be smaller than the second one, you provided {split}'
            result.append(rsna(elt.split('_')[0], split, is_train))
        else:
            assert elt.lower() in dataset_options.keys(), f'The chosen dataset "{elt}" is not existing, please provide a valid option: \n {list(dataset_options.keys())}'
            result.append(dataset_options[elt.lower()]())
    return result
def checkSize(dataset):
        if 'imagenet' in dataset or 'rsna' in dataset:
            image_size = 224
        else:
            if not args.audio:
                raise NotImplementedError(f'Image size for dataset {dataset} not implemented')
        return image_size

if args.training_dataset != "":
    try:
        eval(args.training_dataset)
        trainSet = prepareDataLoader(eval(args.training_dataset), is_train=True)
        if args.training_image_size == -1:
            args.training_image_size = checkSize(eval(args.training_dataset)[0])
    except NameError:
        trainSet = prepareDataLoader(args.training_dataset, is_train=True)
        if args.training_image_size == -1:
            args.training_image_size = checkSize(args.training_dataset)
else:
    trainSet = []
if args.validation_dataset != "":
    try:
        eval(args.validation_dataset)
        validationSet = prepareDataLoader(eval(args.validation_dataset), is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(eval(args.validation_dataset)[0])
    except NameError:
        validationSet = prepareDataLoader(args.validation_dataset, is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(args.validation_dataset)
else:
    validationSet = []

if args.test_dataset != "":
    try:
        eval(args.test_dataset)
        testSet = prepareDataLoader(eval(args.test_dataset), is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(eval(args.test_dataset)[0])
    except NameError:
        testSet = prepareDataLoader(args.test_dataset, is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(args.test_dataset)
else:
    testSet = []

print(" dataloaders,", end='')
