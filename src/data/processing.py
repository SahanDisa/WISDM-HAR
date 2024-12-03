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

directory = './data/wisdm-dataset/raw/'
column_names = ['ID', 'activity', 'timestamp', 'x', 'y', 'z']
sequence_length = 100

label_encoder = LabelEncoder()

def process_file(file_path):
    df = pd.read_csv(file_path, header=None, names=column_names)
    df['z'] = df['z'].astype(str).str.replace(r';$', '', regex=True).astype(float)
    # df = df[df['activity'].isin(['A', 'B', 'C', 'D', 'E', 'F'])] # slight change on feature selection
    df['activity'] = label_encoder.fit_transform(df['activity'])
    return df

def create_sequences(df, seq_length=sequence_length):
    sequences = []
    labels = []
    
    for start in range(0, len(df) - seq_length + 1, seq_length):
        seq = df[['x', 'y', 'z']].iloc[start:start + seq_length].values
        label = df['activity'].iloc[start]
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)


def data_processing():
    all_sequences = []
    all_labels = []

    # Process each file and create sequences
    for root, dirs, files in tqdm(os.walk(directory), desc="Processing Files"):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                df = process_file(file_path)
                sequences, labels = create_sequences(df, sequence_length)
                all_sequences.extend(sequences)
                all_labels.extend(labels)

    # Convert lists to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)

    # Train-Validation-Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(all_sequences, all_labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors
    # X_train, X_val, X_test = map(torch.tensor, (X_train, X_val, X_test))
    # y_train, y_val, y_test = map(torch.tensor, (y_train, y_val, y_test))
    X_train, X_val, X_test = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(X_test) 
    y_train, y_val, y_test =torch.from_numpy(y_train) , torch.from_numpy(y_val), torch.from_numpy(y_test)

    return all_sequences, all_labels, X_train, X_val, X_test, y_train, y_val, y_test


# Define a PyTorch Dataset class
class ActivityDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

