#include <stdio.h>
#include <cuda_runtime.h>

void writeToCSV(const char *filename, int dataSize, float globalTime, float sharedTime) {
    FILE *file = fopen(filename, "a"); /
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    fprintf(file, "%d, %f, %f\n", dataSize, globalTime, sharedTime); 
    fclose(file);
}
       

__global__ void copyGlobalToShared(float *globalData, float *result, int size) {
    __shared__ float sharedData[256];  

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
        sharedData[threadIdx.x] = globalData[idx];  
        __syncthreads();
        result[idx] = sharedData[threadIdx.x];     
    }
}

int main() {
    const char *filename = "memoryMigrationTime.csv";
    FILE *file = fopen(filename, "w"); 
    if (file != NULL) {
        fprintf(file, "Data Size (MB),CPU -> GPU Global (ms),GPU Global -> Shared (ms)\n");
        fclose(file);
    }

    int dataSize = 1 << 20; // = 1MB, bitshift 
    int steps = 2000; 
    int numElements;

    for (int i = 0; i <= steps; i += 10) {
        dataSize = i * (1 << 20); 
        numElements = dataSize / sizeof(float);

        float *A = (float *)malloc(dataSize);
        float *B = (float *)malloc(dataSize);
        float *d_A, *d_B;

        cudaMalloc((void **)&d_A, dataSize);
        cudaMalloc((void **)&d_B, dataSize);

        for (int j = 0; j < numElements; j++) {
            A[j] = 1.0f;
        }

        // CPU -> GPU Global Memory
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaMemcpy(d_A, A, dataSize, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float globalTime;
        cudaEventElapsedTime(&globalTime, start, stop);

        // GPU Global -> Shared Memory
        dim3 blockDim(256);
        dim3 gridDim((numElements + blockDim.x - 1) / blockDim.x);

        cudaEventRecord(start);

        copyGlobalToShared<<<gridDim, blockDim>>>(d_A, d_B, numElements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float sharedTime;
        cudaEventElapsedTime(&sharedTime, start, stop);

        writeToCSV(filename, dataSize / (1 << 20), globalTime, sharedTime);

        free(A);
        free(B);
        cudaFree(d_A);
        cudaFree(d_B);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Measurements saved to %s\n", filename);
    return 0;
}


