#imports
from turtle import forward
import torch
import torch.nn as nn
from torch.functional import F

#Structure of Siamese 
class Subsampling(nn.Module):
    '''Subsampling operation, it is done by convolving over stride with the same as kernel size such 
    that there is no overlapping parts'''
    def __init__(self, input_channels, kernel_size = 2):
        super(Subsampling, self)
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride= kernel_size)
    def forward(self, x):
        return self.conv(x)

class Siamese(nn.Module):
    '''Siamese module'''
    def __init__(self):
        super(Siamese, self).__init__()
        self.main = nn.Sequential(
                    nn.Conv2d(3 ,32, 15, stride = 1 ),
                    Subsampling(32),
                    nn.Conv2d(32, 64, 8, stride = 1),
                    Subsampling(64, 3), 
                    nn.Conv2d(64, 256, 5, stride=1),
                    Subsampling(256)

        )
    
    def forward(self, x):
        return self.main(x)

class Transformation(nn.Module):
    pass
