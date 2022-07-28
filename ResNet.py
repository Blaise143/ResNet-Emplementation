import pandas as pd
import torch, torchvision
import torch.nn as nn
import torch.functional as F

class IdentityBlock(nn.Module):

    def __init__(self,  filters, kernel_size, f,in_channels=3):
        '''
        :param X: data
        :param in_channels: Number of channels from previous input
        :param filters: Number of filters
        :param kernel_size: size of kernel
        '''
        super().__init__(IdentityBlock, self)

        f1, f2, f3 = filters

        self.X_shortcut = X

        self.main_route = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=f1, kernel_size=1),
            nn.BatchNorm2d(num_features=f1),
            nn.ReLU(),

            nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=f, padding=0),
            nn.BatchNorm2d(num_features=f2),
            nn.ReLU(),

            nn.Conv2d(in_channels=f2, out_channels=f3, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=f3),
        )

        self.connection = X_shortcut + self.main_route
        self.final_activation = nn.ReLU()

    def forward(self, X):
        X = self.main_route(X)
        X = self.connection(X)
        X = self.final_activation(X)
        return X

trial = IdentityBlock((4,5,6), 3, 3,3)