import matplotlib.pyplot as plt

def plot_loss_curve(history, epochs):
    train_loss = history["train_loss"]
    test_loss = history["test_loss"]

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
