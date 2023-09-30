import train_arguments
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
import numpy as np
import json
import sys
import torch.nn as nn
import torch.nn.functional as F


#run the following on terminal to download the dataset
#wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar 
import tarfile


# Define the path to the downloaded tar file
tar_file = 'images.tar'

# Extract the contents of the tar file
with tarfile.open(tar_file, 'r') as tar:
    tar.extractall()

#################################################
##The following to to create training and validation split folder and arrange the data for processing

import os
import shutil

# Define paths for the extracted dataset and the target directories
dataset_dir = 'Images'
train_dir = 'train'
valid_dir = 'valid'

# Create training and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Define the percentage of images to be in the validation set
validation_split = 0.2  # Adjust as needed

# Iterate through each class directory in the dataset
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_dir)
    num_validation = int(len(images) * validation_split)
    validation_images = images[:num_validation]
    train_images = images[num_validation:]

    # Create class directories in the training and validation directories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

    # Move images to the appropriate directories
    for img in validation_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(valid_dir, class_name, img)
        shutil.move(src, dst)

    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.move(src, dst)
 #examine info

