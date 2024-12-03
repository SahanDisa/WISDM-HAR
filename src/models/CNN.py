import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils import weight_norm
from torchinfo import summary
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Calculate the flattened dimension after Conv2D layers
        self.flatten_dim = 128 * (100 // (2 * 2 * 2)) * (3 // 1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu_fc(self.fc1(x)))
        x = self.fc2(x)
        
        return x
