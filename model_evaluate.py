import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import warnings

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation function for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])  # Extract the last time step output
        
        # Apply sigmoid activation to get probability
        return self.sigmoid(out)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.simplefilter("ignore")
# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1

model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
checkpoint_path = 'checkpoint.pth'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("checkpoint loaded")
else:
    raise FileNotFoundError("checkpoint doesnt exist")

def pad_sequences(sequences, max_len):
    padded = torch.zeros((len(sequences), max_len, 1))  
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length, 0] = seq[:length, 0]
    return padded

model.eval()

sample_curve_path = r"rcb data\gaia rcbs 1\0.85jds noisey1_new_Gaia20fda.csv"
sample_curve = pd.read_csv(sample_curve_path)
sample_curve = sample_curve.sort_values(by="avg Mag")

# Ensure the column contains only valid numeric data
mags = pd.to_numeric(sample_curve["avg Mag"], errors='coerce').dropna()

# Check if the column is empty after cleaning
if mags.empty:
    raise ValueError("The 'avg Mag' column is empty or contains only invalid data.")

# Convert to PyTorch tensor and add batch and feature dimensions
mags_tensor = torch.tensor(mags.to_numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

with torch.no_grad():
    probability = model(mags_tensor).item()

print(f"Probability of being an RCB star: {probability:.4f}")