import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':

    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    for size in sizes:
        # Change size to match the filename from the C++ program
        filename = f"timing_results_size_{size}.csv"

        # Read the CSV file
        df = pd.read_csv(filename)

        # Reshape data for seaborn
        df_melted = df.melt(id_vars=["Run"],
                            value_vars=["CPU Time (ms)", "GPU Atomic Time (ms)", "GPU Reduction Time (ms)"],
                            var_name="Method",
                            value_name="Time (ms)")

        # Create a violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Method", y="Time (ms)", data=df_melted, inner="point", palette="muted")

        # Add titles and labels
        plt.title(f"Execution Time Distribution (Array Size = {size})", fontsize=14)
        plt.xlabel("Computation Method", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)

        # Show the plot
        plt.savefig(f'violin_plot_size{size}.pdf')
        plt.show()
