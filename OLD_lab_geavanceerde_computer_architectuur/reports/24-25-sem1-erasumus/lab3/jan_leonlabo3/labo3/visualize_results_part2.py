import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
file_path_1 = r"C:\\Users\\janma\\OneDrive - KU Leuven\\Geavanceerde computerarchitecturen\\labo\\labo3\\threads_per_block-execution_time.csv"
file_path_2 = r"C:\\Users\\janma\\OneDrive - KU Leuven\\Geavanceerde computerarchitecturen\\labo\\labo3\\threads_per_block-execution_time_divergence.csv"
file_path_3 = r"C:\\Users\\janma\\OneDrive - KU Leuven\\Geavanceerde computerarchitecturen\\labo\\labo3\\threads_per_block-execution_time_divergence_optimised.csv"

data1 = pd.read_csv(file_path_1, sep=';')
data2 = pd.read_csv(file_path_2, sep=';')
data3 = pd.read_csv(file_path_3, sep=';')

# Clean column names by stripping whitespace
data1.columns = data1.columns.str.strip()
data2.columns = data2.columns.str.strip()
data3.columns = data3.columns.str.strip()

# Print out the column names to verify
print("Column names:", data1.columns)
print("Column names:", data2.columns)
print("Column names:", data3.columns)

# Extract columns from the first file
threads_per_block_1 = data1['ThreadsPerBlock'].values
kernel_times_1 = data1['Kernel (ms)'].values

# Extract columns from the second file
threads_per_block_2 = data2['Threads/Block'].values
kernel_times_2 = data2['Kernel (ms)'].values

# Extract columns from the third file
threads_per_block_3 = data3['Threads/Block'].values
kernel_times_3 = data3['Total (ms)'].values

# Create a figure for the plot
plt.figure(figsize=(10, 6))

plt.xticks(np.arange(min(threads_per_block_1), max(threads_per_block_1)+1, step=100))  # Adjust the step size as needed
plt.gca().set_facecolor('white')
plt.xticks(rotation=45, fontsize=8)
plt.style.use('default')

# Plot kernel times vs threads per block for the first dataset
plt.plot(threads_per_block_1, kernel_times_1, marker='o', color='b', label="Kernel Time (Original)")

# Plot kernel times vs threads per block for the second dataset (divergence)
plt.plot(threads_per_block_2, kernel_times_2, marker='x', color='r', label="Kernel Time (With Divergence)")

# Plot kernel times vs threads per block for the third dataset (divergence optimised)
plt.plot(threads_per_block_3, kernel_times_3, marker='x', color='g', label="Kernel Time (With Divergence Optimised)")

# Add labels and title
plt.xlabel("Threads per Block")
plt.ylabel("Kernel Time (ms)")
plt.title("Kernel Execution Time vs Threads per Block")

# Set y-axis to logarithmic scale
plt.yscale('log')

# Add a legend to distinguish between the two datasets
plt.legend()

# Add a grid for better readability
plt.grid(True)

# Show the plot
plt.show()
