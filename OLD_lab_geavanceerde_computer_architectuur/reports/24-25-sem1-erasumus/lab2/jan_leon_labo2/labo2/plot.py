import matplotlib.pyplot as plt

# Gegeven data
# aantal_elementen = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
# cpu = [0, 0, 3, 13, 155, 1321, 13145, 168352, 1558857]
# gpu_reduction = [223, 173, 362, 240, 399, 325, 1006, 8486, 42132]
# gpu_atomic = [42, 43, 34, 33, 76, 101, 455, 2154, 20751]

# aantal_elementen = [10, 100, 1000, 10000, 100000, 1000000]
# cpu = [0, 0, 3, 13, 155, 1321]
# gpu_reduction = [223, 173, 362, 240, 399, 325]
# gpu_atomic = [42, 43, 34, 33, 76, 101]

aantal_elementen = [10000, 100000, 1000000, 10000000, 100000000, 1000000000]
cpu = [13, 155, 1321, 13145, 168352, 1558857]
gpu_reduction = [240, 399, 325, 1006, 8486, 42132]
gpu_atomic = [33, 76, 101, 455, 2154, 20751]

# Maak de plot
plt.figure(figsize=(10, 6))

# Plot de data
plt.plot(aantal_elementen, cpu, label='CPU (microsecond)', marker='o')
plt.plot(aantal_elementen, gpu_reduction, label='GPU_Reduction (microsecond)', marker='o')
plt.plot(aantal_elementen, gpu_atomic, label='GPU_Atomic (microsecond)', marker='o')

# Logaritmische schaal voor de x-as (aantal elementen)
plt.xscale('log')

# Titels en labels
plt.title('Vergelijking van CPU, GPU Reduction en GPU Atomic tijd per aantal elementen')
plt.xlabel('Aantal elementen')
plt.ylabel('Tijd (microseconden)')
plt.legend()

# Toon de grafiek
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Opslaan als PNG in de huidige directory
plt.savefig('labo2/vergelijking_gpu_cpu.png', format='png')

# Sluit de plot om geheugen vrij te geven
plt.close()

#------------------------------------------------------------------------------------------------------------

# Bereken de tijd per element (in microseconden) voor elke methode
cpu_per_element = [t / e if e != 0 else 0 for t, e in zip(cpu, aantal_elementen)]
gpu_reduction_per_element = [t / e if e != 0 else 0 for t, e in zip(gpu_reduction, aantal_elementen)]
gpu_atomic_per_element = [t / e if e != 0 else 0 for t, e in zip(gpu_atomic, aantal_elementen)]

# Maak de plot
plt.figure(figsize=(10, 6))

# Plot de data voor tijd per element
plt.plot(aantal_elementen, cpu_per_element, label='CPU (Tijd per element)', marker='o')
plt.plot(aantal_elementen, gpu_reduction_per_element, label='GPU_Reduction (Tijd per element)', marker='o')
plt.plot(aantal_elementen, gpu_atomic_per_element, label='GPU_Atomic (Tijd per element)', marker='o')

# Logaritmische schaal voor de x-as (aantal elementen)
plt.xscale('log')

# Titels en labels
plt.title('Vergelijking van CPU, GPU Reduction en GPU Atomic tijd per element')
plt.xlabel('Aantal elementen')
plt.ylabel('Tijd per element (microseconden)')
plt.legend()

# Toon de grafiek
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Opslaan als PNG in de huidige directory
plt.savefig('labo2/vergelijking_gpu_cpu_per_element.png', format='png')

# Sluit de plot om geheugen vrij te geven
plt.close()
