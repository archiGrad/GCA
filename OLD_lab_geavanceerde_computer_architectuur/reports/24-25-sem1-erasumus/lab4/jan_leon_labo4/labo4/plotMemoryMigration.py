import pandas as pd
import matplotlib.pyplot as plt

filename = "memoryMigrationTime.csv"

data = pd.read_csv(filename)

data_size = data["Data Size (MB)"]
cpu_to_gpu_time = data["CPU -> GPU Global (ms)"]
gpu_to_shared_time = data["GPU Global -> Shared (ms)"]

plt.figure(figsize=(10, 6))
plt.plot(data_size, cpu_to_gpu_time, label="CPU -> GPU Global", marker='o')
plt.plot(data_size, gpu_to_shared_time, label="GPU Global -> Shared", marker='s')

plt.title("Memory Migration Times")
plt.xlabel("Data Size (MB)")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(True)

plt.savefig("memoryMigrationPlot.png")  # Optioneel: Sla de plot op als PNG
plt.show()
