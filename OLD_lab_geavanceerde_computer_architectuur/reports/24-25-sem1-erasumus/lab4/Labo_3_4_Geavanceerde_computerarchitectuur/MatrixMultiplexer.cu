#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;
using namespace chrono;
#define M 16  
#define K 16  
#define N 16 

__constant__ int d_M = M;
__constant__ int d_K = K;
__constant__ int d_N = N;
__constant__ int const_B[16][16];  

__global__ void matMulGlobalOnly(int* A, int* B, int* C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < d_M) && (col < d_N)){
        int sum = 0;
        for (int k = 0; k < K; ++k) {
            sum = sum +(A[(row * d_K) + k] * B[(k * d_N) + col]);
        }
        C[(row * d_N) + col] = sum;
    }
}

// Shared memory kernel
__global__ void matMulSharedMemory(int *A, int *B, int *C) {
    __shared__ int shared_A[16][16];  // Size depends on block size
    __shared__ int shared_B[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (int i = 0; i < d_K / blockDim.x; ++i) {
        shared_A[threadIdx.y][threadIdx.x] = A[row * d_K + i * blockDim.x + threadIdx.x];
        shared_B[threadIdx.y][threadIdx.x] = B[(i * blockDim.x + threadIdx.y) * d_N + col];
       
        __syncthreads();  // Ensure all threads have loaded their data before computation

        for (int k = 0; k < blockDim.x; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();  // Ensure synchronization before loading next tiles
    }

    if (row < d_M && col < d_N) {
        C[row * d_N + col] = sum;
    }
}

// Constant memory kernel
__global__ void matMulConstantMemory(int* A, int *C) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < d_M && col <  d_N) {
        for(int k= 0; k< d_K; k++){
            sum += A[row*d_K+k]* const_B[k][col];
        }
        C[row*d_N+col] = sum;
    }
}

void printMatrix(int *matrix, int rows, int col){
    for(int i=0; i<rows;i++){
        for(int j = 0; j < col; j++){
            std::cout << matrix[i*col+j]<<" ";
        }
    }
    std::cout << std::endl;
}

void matrixMultiplication() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(int);
    size_t size_B = K * N * sizeof(int);
    size_t size_C = M * N * sizeof(int);
   

    A = (int*)malloc(M*K*sizeof(int));
    B = (int*)malloc(K*N*sizeof(int));
    C = (int*)malloc(M*N*sizeof(int));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = 1;  
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = 2;  
        }
    }

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(const_B, B, size_B);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Timing for Global Only Kernel
    auto start = high_resolution_clock::now();
    matMulGlobalOnly<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    cout << "Global-only memory kernel time: " << duration.count() << " microseconds" << endl;
    printMatrix(C,M,N);

    // Timing for Shared Memory Kernel
    start = high_resolution_clock::now();
    matMulSharedMemory<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    cout << "Shared memory kernel time: " << duration.count() << " microseconds" << endl;
    printMatrix(C,M,N);

    // Timing for Constant Memory Kernel
    start = high_resolution_clock::now();
    matMulConstantMemory<<<numBlocks, threadsPerBlock>>>(d_A, d_C);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    cout << "Constant memory kernel time: " << duration.count() << " microseconds" << endl;
    printMatrix(C,M,N);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
}

int main() {
    matrixMultiplication();
    return 0;
}