from args import args
import torchvision
import json
import os
from torchvision import transforms, datasets
import torch 
import numpy as np
from PIL import Image
import scipy.io
import tqdm
from collections import defaultdict
import pandas as pd


available_datasets = os.listdir(args.dataset_path)
print('Available datasets:', available_datasets)
all_results = defaultdict(dict)

# Generate data for iNaturalist
# if folder does not exist, create it:
path = os.path.join(args.dataset_path, 'rsna', 'train_images_processed_512')
if os.path.exists(path):
    result = {"data":[], "targets":[], "name":"rsna512", "num_classes":2, "name_classes":["no cancer", "cancer"], "num_elements_per_class":[], "classIdx":{}}
    # read csv file
    df = pd.read_csv(os.path.join('datasets', 'rsna', 'train.csv'))
    result["num_elements_per_class"].append(len(df[df['cancer'] == 0]))
    result["num_elements_per_class"].append(len(df[df['cancer'] == 1]))
    for i, row in df.iterrows():
        row['site_id'],row['patient_id'],row['image_id'],row['laterality'],row['view'],row['age'],row['implant'],row['machine_id']
        new_image_path = f"{row['patient_id']}_row['site_id']_{row['age']}_{row['implant']}"
        # read the four images in the folder:
        #for image in os.listdir(os.path.join(path, row['image_id'])):
            ##### If processing required, otherwise just ignore this part
            # # read the image
            # img = Image.open(os.path.join(path, row['image_id'], image))
            # # save the image
            # img.save(os.path.join(args.dataset_path, new_path, new_image_path, image))
        result["data"].append(os.path.join("rsna", "train_images_processed_512", str(row['patient_id']), str(row['image_id']) + '.png'))
        result["targets"].append(row['cancer'])
    all_results['rsna512'] = result
    print("Done for rsna512 with " + str(result["num_classes"]) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")

f = open(args.dataset_path + "datasets_rsna.json", "w")
f.write(json.dumps(all_results))
f.close()
