import torch
import torch.nn as nn
import torchio as tio
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
spatial_transform = tio.Compose([
    tio.Resample(target=(2, 2, 2)),  # Downsample to ~(120, 120, 77)
    tio.CropOrPad(target_shape=(128, 128, 80)),  # Then pad to even size
    tio.ZNormalization()
])

x = torch.rand(size=(2, 240, 240, 155))
subject = tio.Subject(
    image=tio.ScalarImage(tensor=x)  # Shape (1, 2, 240, 240, 155)
)
transformed = spatial_transform(subject)
print(f"The shape of the transformed object is :{transformed.shape}")
x = transformed.image.data
print(f"X's shape: {x.shape}")  # Shape (1, 2, 128, 128,, 80)?
