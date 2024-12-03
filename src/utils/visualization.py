import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def plot_train(model, train_losses, val_losses):
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f'train_val_loss_{model}.png')
    # plt.show()

def plot_confusion_matrix(model, y_true, y_pred):
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
    plt.savefig(f'confusion_matrix_{model}.png')
    # plt.show()