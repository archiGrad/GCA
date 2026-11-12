#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define MATRIX_SIZE 127 

// Constant memory for matrix B
__constant__ int d_B_const[MATRIX_SIZE * MATRIX_SIZE];

// Initialize matrix A with given data
void initMatrix(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        int getal = rand() % 23 ;
        matrix[i] = getal;
    }
}

// Global memory only kernel for integer matrix multiplication
__global__ void matrixMulGlobal(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMulShared(int* A, int* B, int* C, int N) {
    __shared__ int s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    int sum = 0;
    
    // Loop over matrix tiles
    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; m++) {
        // Load tiles into shared memory
        if (row < N && m * BLOCK_SIZE + tx < N)
            s_A[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        else
            s_A[ty][tx] = 0;
        
        if (col < N && m * BLOCK_SIZE + ty < N)
            s_B[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0;
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Constant memory kernel
__global__ void matrixMulConstant(int* A, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * d_B_const[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = MATRIX_SIZE;
    const size_t size = N * N * sizeof(int);

    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);

    initMatrix(h_A, N);
    initMatrix(h_B, N);

   int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // Time global memory version
    cudaEventRecord(start);
    matrixMulGlobal<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Global memory version: %.3f ms\n", milliseconds);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result matrix C
    /*printf("Result matrix Global:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }*/
    
    std::cout << "-------------------------------------------------------" << std::endl;
    // Time shared memory version
    cudaEventRecord(start);
    matrixMulShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Shared memory version: %.3f ms\n", milliseconds);
    /*printf("Result matrix Shared:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }*/
    
    std::cout << "-------------------------------------------------------" << std::endl;
    //Constant memory version
    cudaMemcpyToSymbol ( d_B_const , h_B , MATRIX_SIZE * MATRIX_SIZE * sizeof ( int ) ) ;
    cudaEventRecord(start);
    matrixMulConstant<<<numBlocks, threadsPerBlock>>>(d_A, d_C, N);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Constant memory version: %.3f ms\n", milliseconds);
    /*printf("Result matrix Constant:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }*/

    // Free allocated memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}