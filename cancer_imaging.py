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
NUM_EPOCHS = 4
LEARNING_RATE = 0.01
BATCH_SIZE = 1

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "cancer_imaging_modelv1.0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#Creating a transform
spatial_transform = tio.Compose([
    tio.CropOrPad(target_shape=(96, 96, 64)), #The original shape is 240, 240, 155
    tio.ZNormalization()
])

#Getting the dataloaders, classes and labels
train_dataloader, valid_dataloader, test_dataloader, channels, labels = data_loader.create_dataloaders(DATASET_PATH, transform=spatial_transform, batch_size=BATCH_SIZE)

#Getting the variables for the model
model = Unet(in_channels=len(channels), num_classes=1).to(device) #Originally num_classes=len(labels), but realize I am outputting one value, so should have it at 1
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

start_time = time.time()
results = engine.train(model=model, 
                       train_dataloader=train_dataloader, 
                       valid_dataloader=valid_dataloader,
                       test_dataloader=test_dataloader, 
                       loss_fn=utils.dice_bce_loss,
                       optimizer=optimizer,
                       epochs=NUM_EPOCHS,
                       device=device)
end_time = time.time()

print(f"The model took: {end_time - start_time:.2f} seconds")

print(results)
utils.plot_loss_curve(history=results, epochs=NUM_EPOCHS)

#Saving the path 
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) #Only saving the state_dict() to save space
print(f"Model saved to {MODEL_SAVE_PATH}")

#Testing on the two samples to confirm our model
SAMPLE1_PATH = Path("sample1/")
SAMPLE2_PATH = Path("sample2/")
utils.random_image_inference(SAMPLE1_PATH, "00495", model, spatial_transform, device)
utils.random_image_inference(SAMPLE2_PATH, "00621", model, spatial_transform, device)

