import torch
import torch.nn as nn
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
