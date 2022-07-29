import pandas as pd
import torch, torchvision
import torch.nn as nn
import torch.functional as F
from torchsummary import summary

class IdentityBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=16) -> None:
        '''
        :param X: data
        :param in_channels: Number of channels from previous input
        :param filters: Number of filters
        :param kernel_size: size of kernel
        '''
        super(IdentityBlock, self).__init__()

        self.main_route = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels//8,
                      kernel_size=5, padding=0, stride=2),
            nn.BatchNorm2d(out_channels//8),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels//8, out_channels=out_channels, kernel_size=7, padding=0, stride=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, X):
        '''
        Forward pass through network
        :param X:
        :return:
        '''
        X_shortcut = X
        X = self.main_route(X)
        X = X + X_shortcut
        X = nn.relu(X)
        return X

class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=16) -> None:
        '''
        :param in_channels: Number of channels of incoming image(3 for RGB, could be 1 for grayscale
        :param out_channels: Number of channels after passed through conv block
        '''
        super(ConvolutionBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels // 8,
                      kernel_size=5, padding=0, stride=2),
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels // 8, out_channels=out_channels, kernel_size=7, padding=0, stride=3),
            nn.BatchNorm2d(out_channels)
        )

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=0, stride=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, X):
        X_shortcut = X
        branch = self.branch(X_shortcut)
        X = self.main(X)
        X = X + branch
        X = F.relu(X)
        return X

class BlaiseResNet(nn.Module):
    def  __init__(self, num_classes, in_channels=3, out_channels=4):
        super(BlaiseResNet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, padding=0, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(out_channels)

        )
        self.identity_block1 = IdentityBlock(in_channels=out_channels, out_channels=out_channels*2)
        self.identity_block2 = IdentityBlock(in_channels= out_channels*2, out_channels= out_channels*4)
        self.convblock1 = ConvolutionBlock(in_channels=out_channels*4, out_channels= out_channels*8)
        self.pool1 = nn.MaxPool2d(kernel_size=7)

        self.identity_block3 = IdentityBlock(in_channels=out_channels*8, out_channels= out_channels*4)
        self.identity_block4 = IdentityBlock(in_channels=out_channels*4, out_channels=out_channels*2)
        self.convblock2 = ConvolutionBlock(in_channels=out_channels*2, out_channels= out_channels)
        self.pool2 = nn.AvgPool2d(kernel_size=3)

        self.fully_connected_layer = nn.Linear(in_features=out_channels, out_features=num_classes)

    def forward(self, X):
        X = self.conv_layer(X)
        X = self.identity_block1(X)
        X = self.identity_block2(X)
        X = self.convblock1(X)
        X = self.identity_block3(X)
        X = self.identity_block4(X)
        X = self.convblock2(X)
        X = self.fully_connected_layer(X)
        return X

summary(BlaiseResNet(num_classes=2))