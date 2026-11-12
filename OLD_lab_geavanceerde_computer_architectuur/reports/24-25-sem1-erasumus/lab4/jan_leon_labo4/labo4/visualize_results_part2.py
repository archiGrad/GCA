import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = 'performance_results.csv'
data = pd.read_csv(file_path)

# Convert BlockSize to a uniform format for sorting
data['BlockSize'] = data['BlockSize'].str.replace('x', '').astype(int)

# Unique M,N,K sizes and block sizes
sizes = data[['M', 'N', 'K']].drop_duplicates()
block_sizes = sorted(data['BlockSize'].unique())

# Plotting
for _, size_row in sizes.iterrows():
    size_filter = (data['M'] == size_row['M']) & (data['N'] == size_row['N']) & (data['K'] == size_row['K'])
    subset = data[size_filter].sort_values('BlockSize')

    plt.figure(figsize=(10, 6))
    plt.plot(subset['BlockSize'], subset['GlobalTime'], label='GlobalTime', marker='o')
    plt.plot(subset['BlockSize'], subset['SharedTime'], label='SharedTime', marker='o')
    plt.plot(subset['BlockSize'], subset['ConstantTime'], label='ConstantTime', marker='o')

    plt.title(f"Performance for M,N,K = {size_row['M']},{size_row['N']},{size_row['K']}")
    plt.xlabel('BlockSize (N x N)')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.xticks(block_sizes, [f'{int(bs/100)}x{int(bs/100)}' for bs in block_sizes])
    plt.tight_layout()
    plt.show()
