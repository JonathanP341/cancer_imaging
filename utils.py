from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torchio as tio
import torch.nn as nn
import torch
import os
import Models

def binary_diceloss(pred, target, smooth=1):
    """
    Computes the DICE loss for binary segmentation
    Args:
        pred: Tensor of predictions [Batch_size, H, W, D]
        pred: Tensor of ground truth [Batch Size, H, W, D]
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Loss

    """
    #Applying sigmoid 
    pred_copy = torch.sigmoid(pred)

    pred = pred.squeeze(dim=1) #Removing that channel dimension, now shape is [Batch_size, H, W, D]

    #Calculate the intersection and union
    intersection = torch.sum(pred_copy * target, dim=(1, 2, 3))
    union = torch.sum(pred_copy, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))

    #Compute the dice coefficient
    dice  = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice.mean()

def dice_bce_loss(logits, target, dice_weight=1, bce_weight=0):
    """
    Computes a loss based on a mix of cross entropy loss and dice loss
    Args:
        logits: The logits from the model
        target: The truth labels
    Returns: 
        Scalar loss, 100% dice, 0% cross entropy
    """
    #The logits are in the form [1, 1, 96, 96, 64] => [1, 96, 96, 64]
    #logits_squeezed = logits.squeeze(dim=1)
    #ce = nn.BCEWithLogitsLoss()(logits, target)
    dice = binary_diceloss(logits, target)

    return dice #* dice_weight + bce_weight * ce

def plot_loss_curve(history, epochs):
    train_loss = history["train_loss"]
    test_loss = history["valid_loss"]

    epochs = range(1, 1 + epochs)

    #Plot loss
    plt.figure(figsize=(15, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

def random_image_inference(sample_path, brats_num: str, model, transform, device):
    #With the sample images, we want to covert them to a format we can iterate on, then get the predictions and compare the two using matplotlib
    channels = ["flair", "t1ce"]
    inputs = []
    for type in channels:
        inputs.append(nib.load(os.path.join(sample_path, "BraTS2021_" + brats_num + "_" + type + ".nii.gz")).get_fdata()) #Getting all of the types we want
    x = np.stack(inputs, axis=0)

    #Getting y
    y = nib.load(os.path.join(sample_path, "BraTS2021_" + brats_num + "_seg.nii.gz")).get_fdata()
    y[y >= 1] = 1

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    subject = tio.Subject(
        image = tio.ScalarImage(tensor=x), #This is (Channels, D, H, W)
        label = tio.LabelMap(tensor=y.unsqueeze(0)) #Makes out seg mask  (1, D, H, W)
    )
    transformed = transform(subject)
    x = transformed.image.data.float() #Keeps it the same size and converts it to a float
    y = transformed.label.data.squeeze(0).float() #Removing that first dimension at the beginning and converting it to a long

    model.eval()
    with torch.inference_mode(): #Getting the logits of the model running x and comparing the two
        x = x.unsqueeze(0) #Adding an extra dimension to simulate the batch layer, now shape of [1, 2, 96, 96, 64]
        
        x, y = x.to(device), y.to(device)
        logits = model(x)

        #Preparing the images
        logits = logits.cpu().detach() #[1, 1, 128, 128, 64]
        y = y.cpu().detach() #[128, 128, 64]
        logits = torch.sigmoid(logits)

        print(dice_bce_loss(logits, y.unsqueeze(dim=0)))

        #Getting a slice of both y and logits so we can view it in 2d
        slice_idx = logits.shape[-1] // 2
        pred_slice = logits[0, 0, :, :, slice_idx]
        mask_slice = y[:, :, slice_idx]
        flair_slice = x[0, 0, :, :, slice_idx].cpu()
        t1ce_slice = x[0, 1, :, :, slice_idx].cpu()

        #Show the images
        plt.figure(figsize=(12, 12))

        #Flair
        plt.subplot(2, 2, 1)
        plt.imshow(flair_slice, cmap="gray")
        plt.title("Flair")

        plt.subplot(2, 2, 2)
        plt.imshow(t1ce_slice, cmap="gray")
        plt.title("T1ce")

        plt.subplot(2, 2, 3)
        plt.imshow(pred_slice, cmap="gray")
        plt.title("Prediction")

        plt.subplot(2, 2, 4)
        plt.imshow(mask_slice, cmap="gray")
        plt.title("Mask")

        plt.tight_layout()
        plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device:
        torch.cuda.empty_cache()

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "cancer_imaging_modelv1.0.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    SAMPLE1_PATH = Path("sample1/")

    load_model = Models.Unet(2, 1).to(device)
    load_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    spatial_transform = tio.Compose([
        tio.CropOrPad(target_shape=(96, 96, 64)), #The original shape is 240, 240, 155
        tio.ZNormalization()
    ])
    
    random_image_inference(SAMPLE1_PATH, "00621", load_model, spatial_transform, device)

if __name__ == "__main__":
    main()

"""
#All images are the same size, all (240, 240, 155)
img = nib.load(SAMPLE1_PATH / "BraTS2021_00495_flair.nii.gz")
data = img.get_fdata()
#print(f"Flair: {data.shape}") #<- Shape is (240, 240, 155)


img = nib.load(SAMPLE1_PATH / "BraTS2021_00495_seg.nii.gz")
data = img.get_fdata()
#print(f"Seg: {data.shape}") #<- Shape is (240, 240, 155)

img = nib.load(SAMPLE1_PATH / "BraTS2021_00495_t1.nii.gz")
data = img.get_fdata()
#print(f"T1: {data.shape}") #<- Shape is (240, 240, 155)

img = nib.load(SAMPLE1_PATH / "BraTS2021_00495_t1ce.nii.gz")
data = img.get_fdata()
#print(f"T1CE: {data.shape}") #<- Shape is (240, 240, 155)

img = nib.load(SAMPLE1_PATH / "BraTS2021_00495_t2.nii.gz")
data = img.get_fdata()
#print(f"T2: {data.shape}") #<- Shape is (240, 240, 155)

#Getting a look at the images
flair = nib.load(SAMPLE1_PATH / "BraTS2021_00495_flair.nii.gz")
flair_data = flair.get_fdata()


#Smoothing
fwhm = 4 #This is a Full Width Half Maximum and is the standard way to smooth kernal size in neuroimaging
#This smoothing is similar to gaussian_filter() from scipy.ndimage but has the kernal smoothing
flair_smoothing = image.smooth_img(flair, fwhm)
plotting.plot_img(flair_smoothing, display_mode='z', cmap="gray")
plt.show()

plotting.plot_img(flair, display_mode='z', cmap="gray")
plt.show()

plotting.plot_img(flair, display_mode='z', cmap="gray") #This is a way faster way to see the same thing as below 
#Default display mode gives just an image, can do x to see through the x axis, or mosaic to see a bunch of different views
#I think the easiest display mode to get a good idea is z, goes from bottom to top starting at the brain stem
plt.show()

#Using flair and plotting a bunch of slices in sequential order through the brain
fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = flair_data.shape[2]
step_size = n_slice // n_subplots #Dividing the size of the slice by the num sub plots, 155 / 16 in this case and then rounded down

fig, ax = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(n_subplots // 2, n_slice - 1, step_size)):
    if (idx == 16):
        break
    else:
        ax.flat[idx].imshow(flair_data[:, :, img], cmap="gray")
        ax.flat[idx].axis("off")
plt.tight_layout()
plt.show()

flair = nib.load(SAMPLE1_PATH / "BraTS2021_00495_flair.nii.gz").get_fdata()
seg = nib.load(SAMPLE1_PATH / "BraTS2021_00495_seg.nii.gz").get_fdata()
t1 = nib.load(SAMPLE1_PATH / "BraTS2021_00495_t1.nii.gz").get_fdata()
t1ce = nib.load(SAMPLE1_PATH / "BraTS2021_00495_t1ce.nii.gz").get_fdata()
t2 = nib.load(SAMPLE1_PATH / "BraTS2021_00495_t2.nii.gz").get_fdata()

slice_idx = 80
#plt.imshow(flair[:, :, slice_idx], alpha = 0.5, cmap="gray")
#plt.imshow(t1[:, :, slice_idx], alpha=0.5, cmap="gray")
#plt.imshow(t1ce[:, :, slice_idx], alpha=0.5, cmap="gray")
plt.imshow(t2[:, :, slice_idx], alpha=0.5, cmap="gray")
plt.imshow(seg[:, :, slice_idx], alpha=0.3, cmap="jet")
plt.title("All super imposed")
plt.show()

plt.imshow(flair[:, :, slice_idx], alpha = 0.7, cmap="gray")
plt.imshow(seg[:, :, slice_idx], alpha=0.3, cmap="jet")
plt.title("FLAIR + Segmentation")
plt.show()

plt.imshow(t1[:, :, slice_idx], cmap="gray")
plt.imshow(seg[:, :, slice_idx], alpha=0.3, cmap="jet")
plt.title("T1 + Seg")
plt.show()

plt.imshow(t1ce[:, :, slice_idx], cmap="gray")
plt.imshow(seg[:, :, slice_idx], alpha=0.3, cmap="jet")
plt.title("T1 CE + Seg")
plt.show()

plt.imshow(t2[:, :, slice_idx], cmap="jet")
plt.imshow(seg[:, :, slice_idx], alpha=0.3, cmap="jet")
plt.title("T2 + Seg")
plt.show()
"""
