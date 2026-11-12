import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
file_path = r"C:\\Users\\janma\\OneDrive - KU Leuven\\Geavanceerde computerarchitecturen\\labo\\labo3\\threads_per_block-execution_time.csv"
data = pd.read_csv(file_path, sep=';')

data.columns = data.columns.str.strip()

# Print out the column names to verify
print("Column names:", data.columns)

# Extract columns
threads_per_block = data['ThreadsPerBlock'].values
kernel_times = data['Kernel (ms)'].values
number_of_blocks = data['Blocks'].values

# Create the figure and primary x-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot kernel times vs threads per block
plt.plot(threads_per_block, kernel_times, marker='o', color='b', label="Kernel Time (ms)")
plt.xlabel("Threads per Block")
plt.ylabel("Kernel Time (ms)")
plt.title("Kernel Execution Time vs Threads per Block")
plt.legend()
plt.grid(True)

# Create a secondary x-axis for the number of blocks
def threads_to_blocks(threads):
    return 64 / threads  # Blocks = Total Threads / Threads Per Block

def blocks_to_threads(blocks):
    return 64 / blocks  # Threads Per Block = Total Threads / Blocks

secax = ax1.secondary_xaxis('top', functions=(threads_to_blocks, blocks_to_threads))
secax.set_xlabel("Number of Blocks")

# Set y-axis to logarithmic scale
plt.yscale('log')

# Show the plot
plt.show()
