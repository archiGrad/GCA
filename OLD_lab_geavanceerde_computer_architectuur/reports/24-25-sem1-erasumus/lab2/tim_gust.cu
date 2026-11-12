#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

void maxDetectionCPU(int *array, int size, int* max) {
    for(int i = 0; i < size; i++) { 
        if(array[i] > *max){
            *max = array[i];
        }
    }
    
}

__global__ void maxDetectionGPUKernel(int *array, int* max) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    atomicMax(max, array[idx]);
}

__device__ int maxNumber(int a, int b) {
    return (a > b) ? a : b;
}

__global__ void maxDetectionGPUReduction(int *array, int size) {
    int idx = threadIdx.x;
    
    //extern __shared__ int sharedData[];
    //sharedData[idx] = array[idx];
    //__syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * idx;
        if (index < blockDim.x) {
            array[index] = maxNumber(array[index], array[index + i]);
        }
        __syncthreads();
    }

    /*if (idx == 0) {
        array[0] = sharedData[0];
    }*/
}

int main(int argc, char **argv) {

    if(argc!= 2){
        std::cerr<<"Verkeerd aantal argumenten"<< std::endl;
        return -1;
    }

    //const int size = std::atoi(argv[1]);2
    const int size = 10;
    int inputArray[size] = {};
    for(int i = 0; i < size; i++){
        inputArray[i] =  rand() % 94415;
    }
    for (int i = 0; i < size; ++i) {
        std::cout << inputArray[i] << ", ";
    }
    std::cout << "\n";

    std::cout << "-----------------------------CPU -----------------------" << "\n";

    const auto start = std::chrono::steady_clock::now();

    int max = INT_MIN;
    maxDetectionCPU(inputArray, size, &max);
    std::cout << "The max value is CPU: " << max;
    std::cout << "\n";

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Elapsed time: " << elapsed_ms.count() << " ms\n";

    std::cout << "-----------------------------GPU Kernel -----------------------" << "\n";

    int *inputArrayGPU;
    cudaMalloc((void**)&inputArrayGPU, sizeof(int) * size);
    cudaMemcpy(inputArrayGPU, inputArray, sizeof(int) * size, cudaMemcpyHostToDevice);
    max = INT_MIN;
    int *gpuMax;
    cudaMalloc((void**)&gpuMax, sizeof(int));
    cudaMemcpy(gpuMax, &max, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda);

    dim3 blockSize(size);
    dim3 gridSize(1);

    maxDetectionGPUKernel<<<gridSize, blockSize>>>(inputArrayGPU, gpuMax);

    cudaMemcpy(&max, gpuMax, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuMax);
    cudaFree(inputArrayGPU);
    
    std::cout << "The max value is: " << max;
    std::cout << "\n";

    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    float ms;
    cudaEventElapsedTime(&ms, start_cuda, stop_cuda);

    std::cout << ms << "ms elapsed";
    std::cout << "\n";

    std::cout << "-----------------------------GPU Reduction -----------------------" << "\n";
    
    int *inputArrayGPURed;
    cudaMalloc((void**)&inputArrayGPURed, sizeof(int) * size);
    cudaMemcpy(inputArrayGPURed, inputArray, sizeof(int) * size, cudaMemcpyHostToDevice);

    cudaEvent_t start_cuda_Red, stop_cuda_Red;
    cudaEventCreate(&start_cuda_Red);
    cudaEventCreate(&stop_cuda_Red);
    cudaEventRecord(start_cuda_Red);

    maxDetectionGPUReduction<<<gridSize, blockSize>>>(inputArrayGPURed, size);

    cudaMemcpy(inputArray, inputArrayGPURed, sizeof(int) * size, cudaMemcpyDeviceToHost);

    std::cout << "The max value is: " << inputArray[0];
    std::cout << "\n";

    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda_Red);
    cudaEventSynchronize(stop_cuda_Red);
    ms = 0;
    cudaEventElapsedTime(&ms, start_cuda_Red, stop_cuda_Red);

    std::cout << ms << "ms elapsed";
    std::cout << "\n";

    cudaFree(inputArrayGPURed);


    return 0;
}

//Bijna niets, tienvoud size, 0.2ms bij voor gpu kernel en 0.02 voor reduction. cpu zeker niets

//max 1024 threads per block

//met striding

//per row en dan gwn kolom 0 --> 0,0 hoogste waarde

