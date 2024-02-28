import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, adjust_shape=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.adjust_shape = adjust_shape
        if adjust_shape:
            self.shape_adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.adjust_shape:
            residual = self.shape_adapter(residual)

        out += residual
        out = self.relu(out)
        return out


class ResSect(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResBlock(in_channels, out_channels, stride, adjust_shape=(stride != 1)))

        for _ in range(1, num_blocks):
            self.blocks.append(ResBlock(out_channels, out_channels))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.sect1 = ResSect(32, 32, 3, 1)
        self.sect2 = ResSect(32, 64, 3, 2)
        self.sect3 = ResSect(64, 128, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 10)

        if pretrained:
            self.load_trained_model()

    @staticmethod
    def load_trained_model():
  
        destination = 'model.pth'

        url = "https://drive.google.com/file/d/1yRBJW21puCu_e4xuxxTdfCNph6hQ3YhD/view?usp=share_link"

        gdown.download(url, destination, quiet=False)

        model = ResModel()
        model.load_state_dict(torch.load(
            destination, map_location=torch.device('cpu')))

        return model

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
