#include <iostream>
#include <cuda_runtime.h>
#include <limits> // Voor std::numeric_limits

// CUDA kernel om array om te draaien
__global__ void flipArray(int *d_in, int *d_out, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_out[size - idx - 1] = d_in[idx];
    }
}

int main() {
    const int size = 10000;  // Grotere array voor betere prestatiemetingen
    int h_in[size];
    int h_out[size];

    // Vul de array
    for (int i = 0; i < size; i++) {
        h_in[i] = i;
    }

    int *d_in, *d_out;

    // Alloceren van geheugen op de GPU
    cudaMalloc(&d_in, size * sizeof(int));
    cudaMalloc(&d_out, size * sizeof(int));

    // Input array kopiëren naar device
    cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

    int bestThreadsPerBlock = 0;
    float bestTime = std::numeric_limits<float>::max(); // Hoogste waarde om mee te vergelijken

    // Experimenteren met verschillende threads per block
    for (int threadsPerBlock = 32; threadsPerBlock <= 512; threadsPerBlock++) {
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        // Tijdopname starten
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Kernel lanceren
        flipArray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, size);

        // Tijdopname stoppen
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Resultaten tonen
        std::cout << "Configuratie - Threads per block: " << threadsPerBlock 
                  << ", Blocks per grid: " << blocksPerGrid 
                  << ", Uitvoertijd: " << milliseconds << " ms" << std::endl;

        // Controleer of dit de beste tijd is
        if (milliseconds < bestTime) {
            bestTime = milliseconds;
            bestThreadsPerBlock = threadsPerBlock;
        }

        // Vrijgeven van cuda events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Resultaten tonen van de optimale configuratie
    std::cout << "Optimale threads per block: " << bestThreadsPerBlock 
              << ", met een uitvoertijd van: " << bestTime << " ms" << std::endl;

    // Output array terug kopiëren naar host om te controleren
    cudaMemcpy(h_out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Controleer enkele waarden
    std::cout << "In: ";
    for (int i = 0; i < 10; i++) { // Eerste 10 elementen
        std::cout << h_in[i] << " ";
    }
    std::cout << "..." << std::endl;

    std::cout << "Out: ";
    for (int i = 0; i < 10; i++) { // Eerste 10 elementen van output
        std::cout << h_out[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Vrijgeven van het device-geheugen
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
