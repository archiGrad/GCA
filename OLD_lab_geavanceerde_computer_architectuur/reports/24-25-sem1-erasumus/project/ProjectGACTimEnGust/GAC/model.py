import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Define the autoencoder model (same architecture as used for training)
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

# Load the saved model
def load_model(file_path, input_dim, device="cpu"):
    model = Autoencoder(input_dim=input_dim).to(torch.device(device))
    # Set weights_only=True to only load model weights
    model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
    return model


def save_model_weights_biases(model):
    for idx, layer in enumerate(model.encoder):
        if isinstance(layer, nn.Linear):
            # Save encoder weights
            encoder_weights = layer.weight.data.cpu().numpy()
            np.savetxt(f"encoder_layer_{idx + 1}_weights.csv", encoder_weights, delimiter=";")

            # Save encoder biases
            encoder_biases = layer.bias.data.cpu().numpy()
            np.savetxt(f"encoder_layer_{idx + 1}_biases.csv", encoder_biases, delimiter=";")

    for idx, layer in enumerate(model.decoder):
        if isinstance(layer, nn.Linear):
            # Save decoder weights
            decoder_weights = layer.weight.data.cpu().numpy()
            np.savetxt(f"decoder_layer_{idx + 1}_weights.csv", decoder_weights, delimiter=";")

            # Save decoder biases
            decoder_biases = layer.bias.data.cpu().numpy()
            np.savetxt(f"decoder_layer_{idx + 1}_biases.csv", decoder_biases, delimiter=";")

# Function to calculate the Mean Squared Error (MSE)
def calculate_mse(original_signal, reconstructed_signal):
    """
    Calculates the Mean Squared Error (MSE) between the original and reconstructed signals.

    Parameters:
    - original_signal: A 1D NumPy array representing the original signal.
    - reconstructed_signal: A 1D NumPy array representing the reconstructed signal.

    Returns:
    - mse: The calculated Mean Squared Error.
    """
    if original_signal.shape != reconstructed_signal.shape:
        raise ValueError("Original and reconstructed signals must have the same shape.")
    
    mse = np.mean((original_signal - reconstructed_signal) ** 2)
    return mse

# Function to reconstruct a signal and return the original and reconstructed signals
def create_reconstructed_signal(model, input_signal, device="cpu"):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Reshape input signal to match model's input requirements (batch size of 1)
        input_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).to(device)
        reconstructed_signal = model(input_tensor).cpu().numpy()

    # Flatten for plotting and returning
    original_sample = input_signal.squeeze()
    reconstructed_sample = reconstructed_signal.squeeze()

    return original_sample, reconstructed_sample

# Plot the original vs reconstructed signal for visual inspection
def plot_original_vs_reconstructed(original_sample, reconstructed_sample, title="Original vs Reconstructed"):
    plt.figure(figsize=(10, 5))
    plt.plot(original_sample, label="Original")
    plt.plot(reconstructed_sample, label="Reconstructed")
    plt.legend()
    plt.title(title)
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.show()

# Plot MSE distributions for normal and anomaly data
def plot_mse_distribution(mse_list_normal, mse_list_anomaly):
    plt.figure(figsize=(10, 6))
    plt.hist(mse_list_normal, bins=30, alpha=0.5, label="Normal Data")
    plt.hist(mse_list_anomaly, bins=30, alpha=0.5, label="Anomaly Data")
    plt.legend()
    plt.title("MSE Distribution for Normal and Anomaly Data")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('mse_distribution.png')

def save_signal_to_csv(signal, filename="input_signal.csv"):
    """
    Save a single signal to a CSV file.

    Parameters:
    - signal: The signal data to save, a 1D NumPy array.
    - filename: The name of the file to save the signal to (default is "input_signal.csv").
    """
    rounded_signal = np.round(signal, 15)
    # Convert the signal to a DataFrame (to save as CSV)
    signal_df = pd.DataFrame(rounded_signal)
    
    # Save to CSV
    signal_df.to_csv(filename, index=False, header=False)
    print(f"Signal saved to {filename}")
    return rounded_signal

# Example usage
if __name__ == "__main__":
    # Load ECG data and preprocess
    csv_file_path = "ecg.csv"
    ecg_data = pd.read_csv(csv_file_path)

    # Separate the features (exclude the last column which is the label)
    features = ecg_data.iloc[:, :-1]
    labels = ecg_data.iloc[:, -1]  # Last column (labels)


    # Separate normal data (label == 1)
    normal_data = features[labels == 1]
    print("Number of normal data signals:", len(normal_data))
    anomaly_data = features[labels == 0]
    print("Number of anomaly data signals:", len(anomaly_data))

    scaler = MinMaxScaler()
    scaled_normal_data = scaler.fit_transform(normal_data)
    scaled_anomaly_data = scaler.fit_transform(anomaly_data)

    scaled_normal_data[0] = save_signal_to_csv(scaled_normal_data[0])

    # Load the trained model
    model_path = "ecg_autoencoder_best.pth"
    input_dim = 140  # Number of features
    device = "cpu"  # Change to "cuda" if using a GPU

    model = load_model(model_path, input_dim, device)

    save_model_weights_biases(model)

    mse_list_normal = []  # List to store MSE for each signal
    mse_list_anomaly = []  # List to store MSE for each signal

    # Loop through each signal
    '''for i in range(len(scaled_normal_data)):
        normal_signal = scaled_normal_data[i]  # Convert to NumPy array

        # Get the original and reconstructed signal
        original_sample, reconstructed_sample = create_reconstructed_signal(model, normal_signal, device)

        # Calculate the MSE
        mse = calculate_mse(original_sample, reconstructed_sample)
        
        mse_list_normal.append(mse)  # Append MSE to list

    for i in range(len(scaled_anomaly_data)):
        anomaly_signal = scaled_anomaly_data[i]  # Convert to NumPy array

        # Get the original and reconstructed signal
        original_sample, reconstructed_sample = create_reconstructed_signal(model, anomaly_signal, device)

        # Calculate the MSE
        mse = calculate_mse(original_sample, reconstructed_sample)
        
        mse_list_anomaly.append(mse)  # Append MSE to list

    # Calculate and print the mean MSE for all normal data signals
    mean_mse_normal = np.mean(mse_list_normal)
    print(f"Mean MSE for all normal signals: {mean_mse_normal:.4f}")
    mean_mse_anomaly = np.mean(mse_list_anomaly)
    print(f"Mean MSE for all anomaly signals: {mean_mse_anomaly:.4f}")

    plot_mse_distribution(mse_list_normal, mse_list_anomaly)
    '''
    '''for i in range(min(5, len(normal_data))):  # Plot first 5 normal signals
        normal_signal = normal_data.iloc[i].values
        original_sample, reconstructed_sample = create_reconstructed_signal(model, normal_signal, device)
        plot_original_vs_reconstructed(original_sample, reconstructed_sample, title=f"Normal Signal {i}")

    # Loop through anomaly signals for visualization
    for i in range(min(5, len(anomaly_data))):  # Plot first 5 anomaly signals
        anomaly_signal = anomaly_data.iloc[i].values
        original_sample, reconstructed_sample = create_reconstructed_signal(model, anomaly_signal, device)
        plot_original_vs_reconstructed(original_sample, reconstructed_sample, title=f"Anomaly Signal {i}")
    '''

    normal_signal = scaled_normal_data[0]  # Convert to NumPy array

    for value in normal_signal:
        print(value)

    # Get the original and reconstructed signal
    original_sample, reconstructed_sample = create_reconstructed_signal(model, normal_signal, device)
    
    print("----------------------------")
    for value in reconstructed_sample:
        print(value)

    # Calculate the MSE
    print(calculate_mse(original_sample, reconstructed_sample))