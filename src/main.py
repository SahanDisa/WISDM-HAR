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

import argparse
from config import get_model_config
from data import processing

from models import MLP, CNN, RNN, LSTM, TCN
from utils import visualization

label_encoder = LabelEncoder()

def run_model(model_name, input_dim, hidden_dim, output_dim, num_channels, kernel_size, dropout, sequence_length, device):
    if model_name == "MLP":
        print("Running MLP model...")
        model = MLP.MLP(input_size=input_dim, sequence_length=sequence_length, num_classes=output_dim)
        return model
        # Add MLP-specific code here
    elif model_name == "CNN":
        print("Running CNN model...")
        model = CNN.CNN(input_size=input_dim, num_classes=output_dim)
        return model 
        # Add CNN-specific code here
    elif model_name == "RNN":
        print("Running RNN model...")
        model = RNN.RNN(input_size=input_dim, hidden_size=hidden_dim, num_classes=output_dim)
        return model
        # Add RNN-specific code here
    elif model_name == "LSTM":
        print("Running LSTM model...")
        model = LSTM.LSTM(input_dim, hidden_dim, output_dim)
        return model
        # Add LSTM-specific code here
    elif model_name == "TCN":
        print("Running TCN model...")
        model = TCN.TCN_HAR(input_dim, output_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        return model
        # Add TCN-specific code here
    else:
        print(f"Model '{model_name}' not recognized. Please select from MLP, CNN, RNN, LSTM, TCN.")
        return
    
def train(train_loader, val_loader, epochs, device):
    # Training loop
    # Initialize lists to store loss values
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0

        # Set up tqdm progress bar
        progress_bar = tqdm(train_loader, desc="Training", unit="batch", leave=False)
        
        for batch_x, batch_y in progress_bar:
            # batch_x = batch_x.transpose(1, 2) # Only for TCN
            optimizer.zero_grad()
            # print(batch_x)
            outputs = model(batch_x.float())
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

            avg_loss = running_train_loss / (progress_bar.n + 1)  # n is the current batch index
            progress_bar.set_postfix(loss=avg_loss)

        # Average training loss for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.float())
                val_loss = criterion(outputs, batch_y)
                running_val_loss += val_loss.item()

        # Average validation loss for the epoch
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x.float())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return all_labels, all_preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Selector")
    parser.add_argument("--model", type=str, required=True, help="Specify the model: MLP, CNN, RNN, LSTM, TCN")
    args = parser.parse_args()
    
    # Load configuration for the selected model
    config = get_model_config(args.model)
    print(f"Configuration for {args.model}: {config}")

    all_sequences, all_labels, X_train, X_val, X_test, y_train, y_val, y_test = processing.data_processing()
    
    # Create DataLoaders
    train_dataset = processing.ActivityDataset(X_train, y_train)
    val_dataset = processing.ActivityDataset(X_val, y_val)
    test_dataset = processing.ActivityDataset(X_test, y_test)

    train_loader = processing.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = processing.DataLoader(val_dataset, batch_size=32)
    test_loader = processing.DataLoader(test_dataset, batch_size=32)
    print('Data Loading Completed!')
    
    # Run the selected model
    input_dim = 3 
    hidden_dim = 64
    output_dim = len(np.unique(all_labels))
    num_channels = [16,32,64,128]
    kernel_size=3
    dropout=0.2
    sequence_length=100
    device = 'cpu'
    print(f"Input Size : {input_dim} Output Size : {output_dim}")
    model = run_model(args.model, input_dim, hidden_dim, output_dim, num_channels, kernel_size, dropout, sequence_length, device)
    print(summary(model))

    epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model, train_losses, val_losses = train(train_loader, val_loader, epochs, device)
    print('Training Completed!')

    visualization.plot_train(args.model, train_losses, val_losses)

    # Get predictions and labels
    y_true, y_pred = evaluate_model(model, test_loader)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nCR by library method=\n", classification_report(y_true, y_pred)) 
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R'])
    fig, ax = plt.subplots(figsize=(12,12))
    disp.plot(ax=ax,cmap='Blues', values_format='.0f')
    plt.title("Confusion Matrix for Test Set")
    plt.savefig(f'confusion_matrix_{args.model}.png')
    # plt.show()
