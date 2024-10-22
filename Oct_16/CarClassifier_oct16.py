import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

nHidden = 100
imgSize = 64
nChannels = 3 # TODO Set correct params
nFilters = 32
nClasses = 25

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nChannels,  nFilters*2),
            nn.BatchNorm2d(nFilters * 2),
            nn.ReLU(True),
            nn.Conv2d(nFilters*2,  nFilters*1),
            nn.BatchNorm2d(nFilters * 1),
            nn.MaxPool2d(2,2),   
            nn.Conv2d(nFilters*1,  nFilters*1),
            nn.Linear(nFilters*1, nHidden),
            nn.Linear(nHidden, nClasses),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        return self.layers(x)
        