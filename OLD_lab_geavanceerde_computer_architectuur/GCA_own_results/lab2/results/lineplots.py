import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob  # To find multiple files

# List of file sizes to process
sizes = [16, 32, 64, 128, 256, 512]  # Add sizes as needed

# Create a DataFrame to store results
results = []

# Loop through files for different sizes
for size in sizes:
    filename = f"timing_results_size_{size}.csv"
    try:
        df = pd.read_csv(filename)

        # Compute mean execution time for each method
        cpu_mean = df["CPU Time (ms)"].mean()
        gpu_atomic_mean = df["GPU Atomic Time (ms)"].mean()
        gpu_reduce_mean = df["GPU Reduction Time (ms)"].mean()

        # Append results to the list
        #results.append(["CPU", size, cpu_mean])
        results.append(["GPU Atomic", size, gpu_atomic_mean])
        results.append(["GPU Reduction", size, gpu_reduce_mean])

    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Skipping.")

# Convert results into a DataFrame
df_results = pd.DataFrame(results, columns=["Method", "Size", "Mean Time (ms)"])

# Plot the results
plt.figure(figsize=(10, 6))
sns.lineplot(x="Size", y="Mean Time (ms)", hue="Method", marker="o", data=df_results, palette="muted")

# Log scale for better visualization if needed (optional)
plt.xscale("log")
plt.yscale("log")

# Labels and title
plt.title("Execution Time vs. Array Size", fontsize=14)
plt.xlabel("Array Size", fontsize=12)
plt.ylabel("Mean Execution Time (ms)", fontsize=12)
plt.legend(title="Computation Method")

# Show the plot
plt.savefig("executiontime_vs_input_size_without_cpu.pdf")
plt.show()
