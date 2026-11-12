import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
file_path_1 = r"memory_access_results.csv"

data1 = pd.read_csv(file_path_1, sep=';')

# Clean column names by stripping whitespace
data1.columns = data1.columns.str.strip()

# Print out the column names to verify
print("Column names:", data1.columns)

# Extract columns from the first file
threads_per_block_1 = data1['ThreadsPerBlock'].values
coalesced_times_1 = data1['CoalescedTime(ms)'].values
uncoalesced_times_1 = data1['UncoalescedTime(ms)'].values

# Create a figure for the plot
plt.figure(figsize=(10, 6))

# Plot both coalesced and uncoalesced times on the same axis
plt.plot(threads_per_block_1, coalesced_times_1, marker='o', color='b', label="Coalesced Time (Original)")
plt.plot(threads_per_block_1, uncoalesced_times_1, marker='x', color='r', label="Uncoalesced Time (Original)")

# Add labels and title
plt.xlabel("Threads per Block")
plt.ylabel("Time (ms)")
plt.title("Coalesced vs Uncoalesced Time vs Threads per Block")

# Add a legend to distinguish between the two datasets
plt.legend()

# Add grid for better readability
plt.grid(True)

# Show the plot
plt.show()