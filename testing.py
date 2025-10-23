import torch
import torch.nn as nn
import torchio as tio
import numpy as np
import nibabel as nib
import torchio as tio
from pathlib import Path
import os
import matplotlib.pyplot as plt
"""
x = torch.rand(size=(16, 2, 2, 2))

c2 = nn.ConvTranspose3d(16, 8, 3, 3) #This will shrink the dimensions from 16->8, but at the same time it will triple the height, width and depth
x = c2(x)
print(x.shape)
c2 = nn.ConvTranspose3d(8, 4, 4, 4) #This will shrink dims from 8->4 but 4x spatial dimensions
x = c2(x)
print(x.shape)
c3 = nn.ConvTranspose3d(4, 1, 1, 1) #This will shrinks dims 4->1 but keep spatial dims the same
x = c3(x)
print(x.shape)
c4 = nn.Conv3d(1, 4, 1, 1)
x = c4(x)
print(x.shape)
#c4 = nn.ConvTranspose3d(1, 1, kernel_size=3, stride=1, padding=2) #This will shrinks dims 4->1 but keep spatial dims the same
#x = c4(x)
#print(x.shape)
"""
spatial_transform1 = tio.Compose([
    tio.Resample(target=(2, 2, 2)),  # Downsample to ~(120, 120, 77)
    tio.CropOrPad(target_shape=(128, 128, 80)),  # Then pad to even size
    tio.ZNormalization()
    #tio.RandomFlip(std=0.5, axes=(1, 2, 3))
])
spatial_transform2 = tio.Compose([
    tio.CropOrPad(target_shape=(192, 192, 80)),  # Then pad to even size
    tio.Resample(target=(2, 2, 1)), # Downsample to 96, 96, 80 
    tio.ZNormalization(),
    tio.RandomNoise(std=0.05)
])
"""
x = torch.rand(size=(2, 240, 240, 155))
subject = tio.Subject(
    image=tio.ScalarImage(tensor=x)  # Shape (1, 2, 240, 240, 155)
)
transformed = spatial_transform(subject)
print(f"The shape of the transformed object is :{transformed.shape}")
x = transformed.image.data
print(f"X's shape: {x.shape}")  # Shape (1, 2, 128, 128,, 80)?


device = "cuda" if torch.cuda.is_available() else "cpu"
if device:
   torch.cuda.empty_cache()
SAMPLE1_PATH = Path("sample1/")
SAMPLE2_PATH = Path("sample2/")
model = Models.Unet(2, 1)
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "cancer_imaging_modelv1.0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
utils.random_image_inference(SAMPLE1_PATH, "00495", model, spatial_transform, device)
utils.random_image_inference(SAMPLE2_PATH, "00621", model, spatial_transform, device)
"""
SAMPLE1_PATH = Path("sample1/")
SAMPLE2_PATH = Path("sample2/")
inputs = []
channels = ["flair", "t1ce", "t1", "t2"]
for type in channels:
    inputs.append(nib.load(os.path.join(SAMPLE1_PATH, "BraTS2021_00495_" + type + ".nii.gz")).get_fdata())
x = np.stack(inputs, axis=0) 
print(f"X's shape after stacking{x.shape}") # (5, 240, 240, 155)

y = nib.load(os.path.join(SAMPLE1_PATH, "BraTS2021_00495_seg.nii.gz")).get_fdata()
print(f"Y's shape before processing: {y.shape}") # (240, 240, 155)

y = torch.from_numpy(y)
x = torch.from_numpy(x)
subject = tio.Subject(
    image=tio.ScalarImage(tensor=x),
    label = tio.LabelMap(tensor=y.unsqueeze(dim=0))  
)
transformed = spatial_transform2(subject)
x = transformed.image.data
y = transformed.label.data.squeeze(dim=0)
print(f"Transformed shape of x: {x.shape}  and y: {y.shape}") # (5, 128, 128, 80)

slice_idx = x.shape[-1] // 2
flair_slice = x[0, :, :, slice_idx] #Getting the middle slice of flair
t1ce_slice = x[1, :, :, slice_idx] #Getting the middle slice of t1ce
t1_slice = x[2, :, :, slice_idx] #Getting the middle slice of flair
t2_slice = x[3, :, :, slice_idx] #Getting the middle slice of t1ce
seg_slice = y[:, :, slice_idx] #Getting the middle slice of flair


#Show the images
plt.figure(figsize=(12, 12))

plt.subplot(2, 3, 1)
plt.imshow(flair_slice, cmap="gray")
plt.title("Flair")

plt.subplot(2, 3, 2)
plt.imshow(t1ce_slice, cmap="gray")
plt.title("T1ce")

plt.subplot(2, 3, 3)
plt.imshow(t1_slice, cmap="gray")
plt.title("T1")

plt.subplot(2, 3, 4)
plt.imshow(t2_slice, cmap="gray")
plt.title("T2")

plt.subplot(2, 3, 5)
plt.imshow(seg_slice, cmap="gray")
plt.title("Segmentation")

plt.show()