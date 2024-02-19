import torch
import torch.nn as nn
import torch.nn.functional as F

'''
You need to implement:

class ResSect(nn.Module):
    def __init__(self, n_filter, n_residual_blocks, beginning_stride):
        Initialize the sector by creating layers needed
        n_filter: number of filters in the conv layers in the blocks
        n_residual_blocks: number of blocks in this sector
        beginning_stride: the stride of the first conv of the first block in the sector

    def forward(self, x):
        Implement computation performed in the sector
        x: input tensor
        You should return the result tensor
'''

class ResModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.sect1 = ResSect(32, 3, 1)
        self.sect2 = ResSect(64, 3, 2)
        self.sect3 = ResSect(128, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 10)

        if pretrained:
            self.load_trained_model()

    def load_trained_model(self):
        '''
        You need to implement this function to:
            1. download the saved pretrained model from your online location
            2. load model from the downloaded model file
        '''
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.sect1(x)
        x = self.sect2(x)
        x = self.sect3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
