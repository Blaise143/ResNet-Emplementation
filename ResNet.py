import pandas as pd
import torch, torchvision
import torch.nn as nn
import torch.functional as F

class IdentityBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=16):
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
                      kernel_size=7, padding=0, stride=2),
            nn.BatchNorm2d(out_channels//8),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels//8, out_channels=out_channels, kernel_size=7, padding=0, stride=3),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, X):
        X_shortcut = X
        X = self.main_route(X)
        X = X + X_shortcut
        X = nn.relu(X)
        return X

trial = IdentityBlock()
print(trial)