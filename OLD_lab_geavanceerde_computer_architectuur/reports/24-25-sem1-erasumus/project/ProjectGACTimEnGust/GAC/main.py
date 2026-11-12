import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess dataset
file_path = 'ecg.csv'
ecg_data = pd.read_csv(file_path)

# Separate the labels (last column) from the features
labels = ecg_data.iloc[:, -1]  # Last column
features = ecg_data.iloc[:, :-1]  # All columns except the last one

# Filter out only normal data (label == 1)
normal_data = features[labels == 1]

# Filter out anomaly data (label == 0)
anomaly_data = features[labels == 0]

# Normalize the data to [0, 1] range
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(normal_data)
#normalized_data = normal_data
scaled_anomaly_data = scaler.transform(anomaly_data)
#scaled_anomaly_data = anomaly_data
# Convert normalized data to PyTorch tensor
train_tensor = torch.tensor(normalized_data, dtype=torch.float32)

# Create DataLoader for the entire dataset (no train/test split)
batch_size = 128
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 140),
            nn.ReLU(),
            nn.Linear(140, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 140),
            nn.ReLU(),
            nn.Linear(140, input_dim),
            nn.Sigmoid(),  # To match normalized input range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Model setup
input_dim = normal_data.shape[1]  # Number of features (columns excluding the label)
model = Autoencoder(input_dim=input_dim).to(torch.device("cpu"))  # Adjust to "cuda" if using GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement before stopping
best_loss = float('inf')  # Initialize with a very large value
patience_counter = 0  # Counter to track how many epochs since the last improvement

# Training the autoencoder with early stopping
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_data in train_loader:
        data = batch_data[0]  # Extract data from TensorDataset

        # Forward pass
        reconstructed = model(data)
        loss = criterion(reconstructed, data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average training loss for this epoch
    train_loss /= len(train_loader)

    # Print the training loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    # Early stopping check
    if train_loss < best_loss:
        best_loss = train_loss
        patience_counter = 0  # Reset the patience counter if loss improves
        # Save the model if it has improved
        torch.save(model.state_dict(), "ecg_autoencoder_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. No improvement in training loss for {patience} epochs.")
            break

# Load the best model after early stopping
model.load_state_dict(torch.load("ecg_autoencoder_best.pth"))

# Testing the autoencoder after training (using anomaly data for testing)
model.eval()
sample_data_tensor = torch.tensor(scaled_anomaly_data, dtype=torch.float32)

with torch.no_grad():
    sample_data = sample_data_tensor[:5]  # Select first 5 samples for demonstration
    reconstructed_samples = model(sample_data)

# Convert reconstructed samples to NumPy for further inspection
original_samples = sample_data.numpy()
reconstructed_samples = reconstructed_samples.numpy()

# Plotting original vs reconstructed signal
plt.figure(figsize=(10, 5))
plt.plot(original_samples[0], label="Original")
plt.plot(reconstructed_samples[0], label="Reconstructed")
plt.legend()
plt.title("Original vs Reconstructed Signal")
plt.show()
