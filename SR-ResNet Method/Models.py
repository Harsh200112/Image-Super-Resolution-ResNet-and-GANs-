import math
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, channels):
        super().__init__()
        self.conv = nn.Sequential(
            self.block(channels, channels, kernel_size, 1, kernel_size//2),
            nn.PReLU(),
            self.block(channels, channels, kernel_size, 1, kernel_size//2)
        )
        
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )
        return self.layer
    
    def forward(self, x):
        residual = x
        y = self.conv(x)
        output = y + residual
        return output
    
class SubPixelConvBlock(nn.Module):
    def __init__(self, kernel_size, channels, scaling_factor = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels*(scaling_factor**2), kernel_size, padding = kernel_size//2)
        self.pixel_shuffle = nn.PixelShuffle(scaling_factor)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class SR_ResNet(nn.Module):
    def __init__(self, large_kernel_size, small_kernel_size, channels, res_blocks, scaling_factor):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels, large_kernel_size, stride = 1, padding = large_kernel_size//2),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(small_kernel_size, channels) for i in range(res_blocks)]
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, small_kernel_size, stride = 1, padding = 1),
            nn.BatchNorm2d(channels)
        )
        
        n_subPix = int(math.log2(scaling_factor))
        self.subPix_blocks = nn.Sequential(
            *[SubPixelConvBlock(small_kernel_size, channels, scaling_factor) for i in range(n_subPix)]
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, 3, large_kernel_size, stride = 1, padding = large_kernel_size//2),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        residual = x
        y = self.residual_blocks(x)
        y = self.conv2(x)
        y = y + residual
        y = self.subPix_blocks(y)
        y = self.conv3(y)
        return y
    

