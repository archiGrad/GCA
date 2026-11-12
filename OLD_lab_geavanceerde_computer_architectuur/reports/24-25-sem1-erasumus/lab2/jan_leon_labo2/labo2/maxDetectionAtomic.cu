#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>  // Toegevoegd voor CPU-timing

//#define N 1024  // Aantal elementen in de 1D-array, aanpasbaar
//#define N 1024 * 1024  // Aantal elementen in de 1D-array, aanpasbaar
#define N 100000 // Aantal elementen in de 1D-array, aanpasbaar

// CUDA-kernel voor maximumdetectie met atomicMax op een 1D-array
__global__ void maxAtomicKernel(int* input, int* maxVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Controleer of de index binnen de array valt
    if (idx < n) {
        atomicMax(maxVal, input[idx]);
    }
}

int main() {
    // 1. Initialisatie van de 1D-array en variabelen
    int *h_array = new int[N];       // Host array
    int maxCPU = INT_MIN;            // Max waarde berekend op de CPU voor controle
    
    // Vul de 1D-array met willekeurige waarden
    for (int i = 0; i < N; ++i) {
        h_array[i] = rand() % 10000;  // Willekeurige waarden tussen 0 en 9999
    }

    // 2. Start de CPU-timer en bereken maximale waarde op de CPU
    auto startCPU = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        if (h_array[i] > maxCPU) {
            maxCPU = h_array[i];
        }
    }
    auto stopCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpuDuration = stopCPU - startCPU;

    // 3. Allocateer geheugen op de GPU voor de 1D-array en de max-variabele
    int *d_array, *d_max;
    cudaMalloc((void**)&d_array, N * sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(int));
    
    // 4. Kopieer de array naar de GPU
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);
    int minVal = INT_MIN;
    cudaMemcpy(d_max, &minVal, sizeof(int), cudaMemcpyHostToDevice);

    // 5. Start de GPU-kernel en meet de uitvoeringstijd
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    maxAtomicKernel<<<blocks, threadsPerBlock>>>(d_array, d_max, N);
    cudaEventRecord(stop);

    // Kopieer het resultaat terug naar de CPU
    cudaMemcpy(&minVal, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    // 6. Print de resultaten
    std::cout << "Maximale waarde (CPU-berekening): " << maxCPU << std::endl;
    std::cout << "Maximale waarde (GPU met atomicMax): " << minVal << std::endl;
    std::cout << "Uitvoeringstijd CPU (ms): " << cpuDuration.count() << std::endl;
    std::cout << "Uitvoeringstijd GPU (ms): " << gpuDuration << std::endl;

    // 7. Opruimen van geheugen
    cudaFree(d_array);
    cudaFree(d_max);
    delete[] h_array;

    return 0;
}
