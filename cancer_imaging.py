import torch
import time
import os

import torchio as tio

from pathlib import Path

from Models import Unet
import data_loader
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
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
WEIGHT_DECAY = 1e-5
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "cancer_imaging_modelv1.6.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#Creating a transform 
#Need to make a better crop into ressample maybe? Because rn its too small and doesnt zoom in enough so hard to see
spatial_transform = tio.Compose([
    tio.CropOrPad(target_shape=(160, 160, 80)),  # Downsize to (160,160,80) to crop the unnecessary parts
    tio.Resample(target=(2, 2, 1)), # Downsample to 80, 80, 80 from that to make it easier to process 
    tio.ZNormalization(),
    tio.RandomNoise(std=0.05)
])
spatial_transform2 = tio.Compose([
    tio.CropOrPad(target_shape=(192, 192, 80)),  # Then pad to even size
    tio.Resample(target=(2, 2, 1)), # Downsample to 96, 96, 80 
    tio.ZNormalization(),
    tio.RandomNoise(std=0.05),
    tio.RandomAffine()
])

print("------WELCOME TO CANCER IMAGING MODEL TRAINING------")
#Getting the dataloaders, classes and labels
print("Getting dataloaders...")
train_dataloader, valid_dataloader, test_dataloader, channels, labels = data_loader.create_dataloaders(DATASET_PATH, transform=spatial_transform2, batch_size=BATCH_SIZE)

#Getting the variables for the model
print("Creating model...")
model = Unet(in_channels=len(channels), num_classes=1).to(device) #Originally num_classes=len(labels), but realize I am outputting one value, so should have it at 1

#To load the model uncomment this section and 
"""
model = Unet(2, 1)
MODEL_PATH = Path("models")
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "cancer_imaging_modelv1.5.pth")))
model.to(device)
"""
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # Not used yet

print("Starting training...")
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
utils.random_image_inference(SAMPLE1_PATH, "00495", model, spatial_transform2, device)
utils.random_image_inference(SAMPLE2_PATH, "00621", model, spatial_transform2, device)

