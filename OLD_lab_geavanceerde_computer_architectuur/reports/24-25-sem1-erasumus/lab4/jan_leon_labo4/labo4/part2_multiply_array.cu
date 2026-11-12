#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

// CSV File for results
#define CSV_FILE "performance_results.csv"

// Matrix multiplication kernels (same as before)
#define TILE_SIZE 64
#define MAX_CONSTANT_SIZE 64

// Global-only memory kernel
__global__ void matMulGlobalOnly(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Global and shared memory kernel
__global__ void matMulSharedMemory(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            B_shared[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            value += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Global and constant memory kernel
__constant__ float B_const[MAX_CONSTANT_SIZE];

__global__ void matMulConstantMemory(float *A, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B_const[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Function to measure performance and write to CSV
void measurePerformance(float *h_A, float *h_B, float *h_C, int M, int N, int K, int blockDimSize, std::ofstream &csvFile) {
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(blockDimSize, blockDimSize);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Global-only memory kernel
    cudaEventRecord(start);
    matMulGlobalOnly<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeGlobalOnly = 0;
    cudaEventElapsedTime(&timeGlobalOnly, start, stop);

    // Shared memory kernel
    cudaEventRecord(start);
    matMulSharedMemory<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeShared = 0;
    cudaEventElapsedTime(&timeShared, start, stop);

    // Constant memory kernel
    std::cout << "K: " << K << " N: " << N << std::endl;   
    std::cout << "gridDim: " << gridDim.x << " " << gridDim.y << std::endl;
    std::cout << "blockDim: " << blockDim.x << " " << blockDim.y << std::endl;
    cudaMemcpyToSymbol(B_const, h_B, K * N * sizeof(float));
    cudaEventRecord(start);
    matMulConstantMemory<<<gridDim, blockDim>>>(d_A, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeConstant = 0;
    cudaEventElapsedTime(&timeConstant, start, stop);

    // Write to CSV
    csvFile << M << "," << N << "," << K << "," << blockDimSize << "x" << blockDimSize << ","
            << timeGlobalOnly << "," << timeShared << "," << timeConstant << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Open CSV file
    std::ofstream csvFile(CSV_FILE);
    csvFile << "M,N,K,BlockSize,GlobalTime,SharedTime,ConstantTime\n";

    int sizes[] = {256, 512, 1024};
    //int sizes[] = {64, 128, 256};
    int blockDims[] = {16, 32, 64};

    for (int s = 0; s < 3; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        float *h_A = (float *)malloc(M * K * sizeof(float));
        float *h_B = (float *)malloc(K * N * sizeof(float));
        float *h_C = (float *)malloc(M * N * sizeof(float));

        for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
        for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

        for (int b = 0; b < 3; b++) {
            measurePerformance(h_A, h_B, h_C, M, N, K, blockDims[b], csvFile);
        }

        free(h_A);
        free(h_B);
        free(h_C);
    }

    csvFile.close();
    printf("Performance results written to %s\n", CSV_FILE);
    return 0;
}
