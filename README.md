# WISDM-HAR

This project includes the final project implementation of Human Activity Recognition(HAR) using Deep Learning with WISDM dataset. It has 18 human activities for 51 human subjects. More information on the dataset could find [here](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset). Use the follow steps to setup the codebase in a local machine. 

## Packages

```
python==3.12.1
pytorch==2.5.1
torchinfo==1.8.0
scikit-learn=1.5.2
numpy==2.0.2
pandas==2.2.3
skorch==1.0.0
matplotlib==3.9.2
tqdm==4.67.0
```

Install above packages via this command

```pip install pytorch==2.5.1 torchinfo==1.8.0 scikit-learn=1.5.2 numpy==2.0.2 pandas==2.2.3 skorch==1.0.0 matplotlib==3.9.2 tqdm==4.67.0```



## Step 1 : Clone the Repository 

``git clone https://github.com/SahanDisa/WISDM-HAR``

through github CLI, use

``gh repo clone SahanDisa/WISDM-HAR``

## Step 2 : Download the dataset 

* Download the WISDM datset from the following link https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset 

* Extract the downloaded zip and extract wisdm-dataset.zip 

* After extracting, move the ``wisdm-dataset`` directory inside to the data directory of the codebase

## Step 3 : Run the Python Script with Arguments 

``python main.py --models="MLP" --epochs=5`` 

make sure to pass models and epochs as argument for smooth execution

Model Options : MLP, CNN, RNN, LSTM, TCN

## Step 4 : Observe the Plots

After successful execution, code will automatically saves the plots for the training/val loss and confusion matrix