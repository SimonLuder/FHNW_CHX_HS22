import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import wandb

from helper import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pytorch_model_summary import summary


class CNNForecaster(nn.Module):
    def __init__(self, kernel_size=3, pool_size=2, padding=0, conv1_channels = 120, 
                 conv2_channels=120, conv3_channels=120, fc_linear_1=180, dropout=0.5):
        '''Convolutional Net class'''
        super(CNNForecaster, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_channels, kernel_size=kernel_size, padding=padding) 
        self.conv2 = nn.Conv1d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=kernel_size, padding=padding) 
        self.conv3 = nn.Conv1d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=kernel_size, padding=padding) 
        
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=1)
        
        self.fc1 = nn.Linear(in_features=conv3_channels*3, out_features=fc_linear_1)
        self.fc2 = nn.Linear(in_features=fc_linear_1, out_features=1)
        
        self.conv3_channels = conv3_channels
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        '''
        Applies the forward pass
        Args:
            x (torch.tensor): input feature tensor
        Returns:
            x (torch.tensor): output tensor of size num_classes
        '''
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(1152, 1),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        logits = self.network(x)
        return logits

class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(4320, 180),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(180, 1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        logits = self.network(x)
        return logits