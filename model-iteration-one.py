import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define the LSTM Model
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

# Define the Dataset Class
class LightCurveDataset(Dataset):
    def __init__(self, data, labels):
        # Convert data and labels to PyTorch tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape
    
    def __len__(self):
        return len(self.data)  # Number of samples in the dataset
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Return sample and label

# Hyperparameters
input_size = 1  # One feature per timestep (e.g., brightness/magnitude)
hidden_size = 64  # Number of hidden units in LSTM
num_layers = 2  # Number of LSTM layers
output_size = 1  # Binary classification (RCB or not)
batch_size = 32  # Number of samples per batch
learning_rate = 0.001  # Learning rate for optimizer
epochs = 50  # Number of training epochs

# Example Data (Replace with real dataset)
n_samples = 1000  # Number of samples in dataset
time_steps = 50  # Number of time steps per sequence
data = np.random.rand(n_samples, time_steps, input_size)  # Simulated light curves
labels = np.random.randint(0, 2, size=(n_samples,))  # Random binary labels (0 or 1)

# Create dataset and data loader
dataset = LightCurveDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCELoss()  # Binary cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Training Loop
for epoch in range(epochs):
    for batch in dataloader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(x_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagate the error
        optimizer.step()  # Update model parameters
    
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Prediction Example
def predict(model, sample):
    model.eval()  # Set model to evaluation mode
    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)  # Convert input to tensor and add batch dimension
    with torch.no_grad():
        prob = model(sample).item()  # Get probability prediction
    return prob

# Example prediction on a new light curve sample
sample_curve = np.random.rand(time_steps, input_size)  # Generate a random sample curve
probability = predict(model, sample_curve)
print(f'Probability of being an RCB star: {probability:.4f}')