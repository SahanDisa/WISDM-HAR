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

from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

# Model parameters
input_dim = 3 
hidden_dim = 64
output_dim = len(np.unique(all_labels))
num_channels = [16, 32, 64, 128]  # TCN
max_epochs=5

print(f"Input Size : {input_dim} Output Size : {output_dim}")

# Define a PyTorch model
def get_model(num_channels, kernel_size, dropout, lr):
    return NeuralNetClassifier(
        module=TCN_HAR,
        module__input_channels=input_dim,
        module__num_classes=output_dim,
        module__num_channels=num_channels,
        module__kernel_size=kernel_size,
        module__dropout=dropout,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        optimizer__lr=lr,
        max_epochs=max_epochs,
        device='cpu',  # Use GPU if available
    )

param_grid = {
    'module__num_channels': [[16, 32], [16, 64], [16, 32, 64]],  # Different channel configurations
    'module__kernel_size': [2, 3],
    'module__dropout': [0.2, 0.5],
    'optimizer__lr': [1e-3, 1e-4],
}
net = get_model(num_channels=num_channels, kernel_size=2, dropout=0.2, lr=1e-3)

grid = GridSearchCV(
    net,
    param_grid,
    n_jobs=-1,
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    refit=True,  # Refit on the best parameters
    verbose=2,
)

X, y = zip(*[(seq, label) for seq, label in train_loader])  # Unpacks batches
X = torch.cat(X)  # Combine all batches into one tensor
y = torch.cat(y)  # Combine all batches into one tensor

grid_result = grid.fit(X.float(), y)
 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))