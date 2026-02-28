import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate)
        )
    
    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p #We return two layers so that they can work together

class UpSample(nn.Module):
    def __init__(self, in_channels, up_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2) #This doubles the size of the image and halfs the dimensions
        self.conv = DoubleConv(in_channels, up_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1) #We concatenate on the first layer 
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_conv_1 = DownSample(in_channels, 32) #This number could change, this is a LOT of layers but whatever
        self.down_conv_2 = DownSample(32, 64)
        self.down_conv_3 = DownSample(64, 128)

        self.bottle_neck = DoubleConv(128, 256)

        self.up_conv_1 = UpSample(256, 128)
        self.up_conv_2 = UpSample(128, 64)
        self.up_conv_3 = UpSample(64, 32)

        self.out = nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=1) #By this point it will be the same size as the input 
    
    def forward(self, x):
        #print(x.shape) #[B, In Channels, 128, 128, 64] Half all of these values since I reduced the size of the model
        down_1, p1 = self.down_conv_1(x)
        #print(p1.shape) #[B, 64, 64, 64, 32]
        down_2, p2 = self.down_conv_2(p1)
        #print(p2.shape) #[B, 128, 32, 32, 16]
        down_3, p3 = self.down_conv_3(p2)
        #print(p3.shape) #[B, 256, 16, 16, 8]

        return self.out(self.up_conv_3(self.up_conv_2(self.up_conv_1(self.bottle_neck(p3), down_3), down_2), down_1))
        """
        b = self.bottle_neck(p4)
        print(b.shape) #[B, 1024, 8, 8, 4]
        
        up_1 = self.up_conv_1(b, down_4)
        print(up_1.shape) #[B, 512, 16, 16, 8]
        up_2 = self.up_conv_2(up_1, down_3)
        print(up_2.shape) #[B, 256, 32, 32, 16]
        up_3 = self.up_conv_3(up_2, down_2)
        print(up_3.shape) #[B, 128, 64, 64, 32]
        up_4 = self.up_conv_4(up_3, down_1)
        print(up_4.shape) #[B, 64, 128, 128, 64]

        out = self.out(up_4) #[B, Out_Channels(4), 128, 128, 64]

        return out
        """

#My first attempt, this was horrible and didnt learn
class BraTSModelV1(nn.Module):
    def __init__(self, in_shape: int, hidden: int, out_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=in_shape, out_channels=hidden, kernel_size=3, stride=1, padding=0), #Kernel size of 3 => (3, 3, 3) because its 3d
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
            #nn.Conv3d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=0),
            #nn.ReLU()
            #nn.MaxPool3d(kernel_size=2, stride=2) #Starts at [Batch, hidden, 150, 150, 120] -> [Batch, hidden, 73, 73, 58]
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=0), 
            nn.ReLU(),
            #nn.MaxPool3d(kernel_size=2, stride=2) #From [Batch, hidden, 73, 73, 58] -> [Batch, hidden, 34, 34, 27]
            nn.Conv3d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=0),
            nn.ReLU()
        )
        #With this block I need to go back up to the size I want which means we need to do 6 ConvTranpose3ds
        self.block3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=0),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            #nn.ConvTranspose3d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=0),
            #nn.ReLU(),
            nn.ConvTranspose3d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=hidden, out_channels=out_shape, kernel_size=3, padding=0)#,
            #nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        #Will have a basic slower return line for now just to understand whats happening better
        """
        x = self.block1(x) #Gets an [8, 8, 150, 150, 120] size tensor
        print(x.shape) #Prints [8, 8, 144, 144, 114]
        x = self.block2(x)
        print(x.shape) #Prints [8, 8, 138, 138, 108]
        x = self.block3(x)
        print(x.shape) #Prints [8, 8, 144, 144, 114]
        x = self.block4(x)
        print(x.shape) #Prints [8, 4, 150, 150, 120]
        return x
        """
        return self.block4(self.block3(self.block2(self.block1(x))))
        