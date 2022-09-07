import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

from common import *
from dataset import ArrhythmiaDataset

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from torch.utils.tensorboard import SummaryWriter


def main():
    class M5(nn.Module):
        def __init__(self,
                     n_input = 1,
                     n_output = 35,
                     stride = 1,
                     n_channel = 32):
            super().__init__()
            self.conv1 = nn.Conv1d(n_input,
                                   n_channel,
                                   kernel_size = 3,
                                   stride = stride)
            self.bn1 = nn.BatchNorm1d(n_channel)
            self.pool1 = nn.MaxPool1d(2)

            self.conv2 = nn.Conv1d(n_channel,
                                   n_channel,
                                   kernel_size = 3)
            self.bn2 = nn.BatchNorm1d(n_channel)
            self.pool2 = nn.MaxPool1d(2)

            self.conv3 = nn.Conv1d(n_channel,
                                   n_channel,
                                   kernel_size = 3)
            self.bn3 = nn.BatchNorm1d(n_channel)
            self.pool3 = nn.MaxPool1d(3)

            self.conv4 = nn.Conv1d(n_channel,
                                   2 * n_channel,
                                   kernel_size = 3)
            self.bn4 = nn.BatchNorm1d(2 * n_channel)
            self.pool4 = nn.MaxPool1d(3)

            self.conv5 = nn.Conv1d(2 * n_channel,
                                   2 * n_channel,
                                   kernel_size = 3)
            self.bn5 = nn.BatchNorm1d(2 * n_channel)
            self.pool5 = nn.MaxPool1d(3)

            # self.conv6 = nn.Conv1d(2 * n_channel,
            #                        2 * n_channel,
            #                        kernel_size = 3)
            # self.bn6 = nn.BatchNorm1d(2 * n_channel)
            # self.pool6 = nn.MaxPool1d(3)

            self.fc1 = nn.Linear(2 * n_channel,
                                 n_channel)
            self.fc2 = nn.Linear(n_channel,
                                 n_output)

        def forward(self,
                    x):
            # print(f'CONV1 INPUT SHAPE: {x.shape}')
            x = self.conv1(x)
            # print(f'CONV1 OUTPUT SHAPE: {x.shape}')
            x = F.relu(self.bn1(x))
            # print(f'POOL1 INPUT SHAPE: {x.shape}')
            x = self.pool1(x)
            # print(f'POOL1 OUTPUT SHAPE: {x.shape}')
            x = self.conv2(x)
            x = F.relu(self.bn2(x))
            # print(f'POOL2 INPUT SHAPE: {x.shape}')
            x = self.pool2(x)
            # print(f'POOL2 OUTPUT SHAPE: {x.shape}')
            x = self.conv3(x)
            x = F.relu(self.bn3(x))
            # print(f'POOL3 INPUT SHAPE: {x.shape}')
            x = self.pool3(x)
            # print(f'POOL3 OUTPUT SHAPE: {x.shape}')
            x = self.conv4(x)
            # print(f'BATCHNORM4 INPUT SHAPE: {x.shape}')
            x = F.relu(self.bn4(x))
            # print(f'POOL4 INPUT SHAPE: {x.shape}')
            x = self.pool4(x)
            # print(f'POOL4 OUTPUT SHAPE: {x.shape}')
            x = self.conv5(x)
            # print(f'BATCHNORM5 INPUT SHAPE: {x.shape}')
            x = F.relu(self.bn5(x))
            # print(f'POOL5 INPUT SHAPE: {x.shape}')
            x = self.pool5(x)
            # print(f'POOL5 OUTPUT SHAPE: {x.shape}')
            # x = self.conv6(x)
            # # print(f'BATCHNORM6 INPUT SHAPE: {x.shape}')
            # x = F.relu(self.bn6(x))
            # # print(f'POOL6 INPUT SHAPE: {x.shape}')
            # x = self.pool6(x)
            # print(f'POOL6 OUTPUT SHAPE: {x.shape}')
            x = F.avg_pool1d(x,
                             x.shape[-1])
            x = x.permute(0,
                          2,
                          1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x,
                                 dim = 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = M5(n_input = 5,
               n_output = 7)
    model.double().to(device)
    print(model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    n = count_parameters(model)
    print('Number of parameters: %s' % n)


if __name__ == '__main__':
    main()
