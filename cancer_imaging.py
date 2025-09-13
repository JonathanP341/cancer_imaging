from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from pathlib import Path
import time

from nilearn import plotting, image
import nibabel as nib

import data_loader
from Models import Unet
import engine
import utils

def binary_diceloss(pred, target, smooth=1):
    """
    Computes the DICE loss for binary segmentation
    Args:
        pred: Tensor of predictions [Batch_size, Input Channels, H, W, D]
        pred: Tensor of ground truth [Batch Size, H, W, D]
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Loss

    """
    pred_copy = pred.clone()
    
    pred_copy[pred_copy < 0] = 0
    pred_copy[pred_copy > 0] = 1

    #Calculate the intersection and union
    intersection = abs(torch.sum(pred_copy * target, dim=(2, 3, 4)))
    union = abs(torch.sum(pred_copy, dim=(2, 3, 4)) + torch.sum(target, dim=(1, 2, 3)))

    #Compute the dice coefficient
    dice  = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice.mean()

def dice_ce_loss(logits, target, dice_weight=0.75, ce_weight=0.25):
    """
    Computes a loss based on a mix of cross entropy loss and dice loss
    Args:
        logits: The logits from the model
        target: The truth labels
    Returns: 
        Scalar loss, 75% dice, 25% cross entropy
    """
    ce = nn.CrossEntropyLoss()(logits, target)
    dice = binary_diceloss(logits, target)

    return dice * dice_weight + ce_weight * ce

#This is going to be the main file where you actually run the code

#Setting the device to be cuda so we can go fast
device = "cuda" if torch.cuda.is_available() else "cpu"
if device:
    torch.cuda.empty_cache()

#print(torch.cuda.is_available())         # True
#print(torch.version.cuda)                # 12.6
#print(torch.cuda.get_device_name(0))     # NVIDIA GeForce GTX 1660 Ti
#In this case I just did the file stuff manually to make it easier for my first time working with a data set like this

DATASET_PATH = Path("data/")
SAMPLE1_PATH = Path("sample1/")
SAMPLE2_PATH = Path("sample2/")
NUM_EPOCHS = 2
LEARNING_RATE = 0.01
BATCH_SIZE = 1

#Creating a transform
spatial_transform = tio.Compose([
    tio.CropOrPad(target_shape=(96, 96, 64)), #The original shape is 240, 240, 155
    tio.ZNormalization()
])

#Getting the dataloaders, classes and labels
train_dataloader, valid_dataloader, test_dataloader, channels, labels = data_loader.create_dataloaders(DATASET_PATH, transform=spatial_transform, batch_size=BATCH_SIZE)

#Getting the variables for the model
model = Unet(in_channels=len(channels), num_classes=len(labels)).to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

start_time = time.time()
results = engine.train(model=model, 
                       train_dataloader=train_dataloader, 
                       valid_dataloader=valid_dataloader,
                       test_dataloader=test_dataloader, 
                       loss_fn=dice_ce_loss,
                       optimizer=optimizer,
                       epochs=NUM_EPOCHS,
                       device=device)
end_time = time.time()

print(f"The model took: {end_time - start_time:.2f} seconds")

print(results)
utils.plot_loss_curve(history=results, epochs=NUM_EPOCHS)

