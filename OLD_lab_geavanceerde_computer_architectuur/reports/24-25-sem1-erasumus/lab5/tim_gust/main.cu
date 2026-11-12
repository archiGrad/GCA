#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>

__device__ int maxNumber(int a, int b) {
    return (a > b) ? a : b;
}

__device__ int minNumber(int a, int b) {
    return (a < b) ? a : b;
}

__global__ void maxDetectionGPUReduction(int *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 1; i < size; i *= 2) {
        int index = 2 * i * idx;
        if (index < size) {
            array[index] = maxNumber(array[index], array[index + i]);
        }
        __syncthreads();
    }
}

__global__ void minDetectionGPUReduction(int *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 1; i < size; i *= 2) {
        int index = 2 * i * idx;
        if (index < size) {
            array[index] = minNumber(array[index], array[index + i]);
        }
        __syncthreads();
    }
}

__global__ void sumDetectionGPUReduction(int *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 1; i < size; i *= 2) {
        int index = 2 * i * idx;
        if (index < size) {
            array[index] = array[index] + array[index + i];
        }
        __syncthreads();
    }
}

__global__ void prodDetectionGPUReduction(int *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 1; i < size; i *= 2) {
        int index = 2 * i * idx;
        if (index < size) {
            array[index] = (array[index] * array[index + i])/2;
        }
        __syncthreads();
    }
}

void writeToCSV(int blocks, int threads, int size, float time_ms, bool assynchroon) {
    if(assynchroon){
        std::ofstream file("output_a.csv", std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            return;
        }
        file << size << ";" << blocks * threads << ";" << time_ms << "\n";
        file.close();
    }
    else{
        std::ofstream file("output_s.csv", std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            return;
        }
        file << size << ";" << blocks * threads << ";" << time_ms << "\n";
        file.close();
    }
}



int main(int argc, char **argv) {

    if(argc!= 2){
        std::cerr<<"Verkeerd aantal argumenten"<< std::endl;
        return -1;
    }

    //const int size = std::atoi(argv[1]);
    /*const int size = 1024;
    
    for (int i = 0; i < size; ++i) {
        std::cout << inputArray[i] << ", ";
    }*/
    std::cout << "\n";
    float time_synchroon;
    float time_asynchroon;
    std::vector<int> array = {256, 1024};
    for (int size : array) {
        std::vector<int> inputArray(size);
        for(int i = 0; i < size; i++){
            inputArray[i] =  (rand() % 9)+1;
        }
        for(int threads = 32; threads < 1024; threads+= 32){
        //std::cout << "-----------------------------GPU Reduction MAX-----------------------" << "\n";
            time_synchroon = 0;
            time_asynchroon = 0;
            int gem = 250;
            for(int i = 0; i < gem; i++){
                int *inputArrayGPURedMax;
                cudaMalloc((void**)&inputArrayGPURedMax, sizeof(int) * size);
                cudaMemcpy(inputArrayGPURedMax, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

                cudaEvent_t start_cuda_Red, stop_cuda_Red;
                cudaEventCreate(&start_cuda_Red);
                cudaEventCreate(&stop_cuda_Red);
                cudaEventRecord(start_cuda_Red);

                dim3 blockSize = threads;
                dim3 gridSize = (size + threads- 1) / threads;

                maxDetectionGPUReduction<<<gridSize, blockSize>>>(inputArrayGPURedMax, size);
                std::vector<int> nieuwArrayMax(size);

                cudaMemcpy(nieuwArrayMax.data(), inputArrayGPURedMax, sizeof(int) * size, cudaMemcpyDeviceToHost);


                cudaDeviceSynchronize();

                //std::cout << "-----------------------------GPU Reduction Min-----------------------" << "\n";
                
                int *inputArrayGPURedMin;
                cudaMalloc((void**)&inputArrayGPURedMin, sizeof(int) * size);
                cudaMemcpy(inputArrayGPURedMin, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

                minDetectionGPUReduction<<<gridSize, blockSize>>>(inputArrayGPURedMin, size);
                std::vector<int> nieuwArrayMin(size);

                cudaMemcpy(nieuwArrayMin.data(), inputArrayGPURedMin, sizeof(int) * size, cudaMemcpyDeviceToHost);

                cudaDeviceSynchronize();

                //std::cout << "-----------------------------GPU Reduction SUM-----------------------" << "\n";
                
                int *inputArrayGPURedSum;
                cudaMalloc((void**)&inputArrayGPURedSum, sizeof(int) * size);
                cudaMemcpy(inputArrayGPURedSum, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

                sumDetectionGPUReduction<<<gridSize, blockSize>>>(inputArrayGPURedSum, size);
                std::vector<int> nieuwArraySum(size);


                cudaMemcpy(nieuwArraySum.data(), inputArrayGPURedSum, sizeof(int) * size, cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();

                //std::cout << "-----------------------------GPU Reduction PROD-----------------------" << "\n";
                
                int *inputArrayGPURedProd;
                cudaMalloc((void**)&inputArrayGPURedProd, sizeof(int) * size);
                cudaMemcpy(inputArrayGPURedProd, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

                prodDetectionGPUReduction<<<gridSize, blockSize>>>(inputArrayGPURedProd, size);
                std::vector<int> nieuwArrayProd(size);


                cudaMemcpy(nieuwArrayProd.data(), inputArrayGPURedProd, sizeof(int) * size, cudaMemcpyDeviceToHost);

                cudaEventRecord(stop_cuda_Red);
                cudaEventSynchronize(stop_cuda_Red);
                float ms;
                cudaEventElapsedTime(&ms, start_cuda_Red, stop_cuda_Red);
                time_synchroon += ms;

                cudaFree(inputArrayGPURedMax);
                cudaFree(inputArrayGPURedMin);
                cudaFree(inputArrayGPURedSum);
                cudaFree(inputArrayGPURedProd);

                /*std::cout << "The max value is: " << nieuwArrayMax[0];
                std::cout << "\n";

                std::cout << "The min value is: " << nieuwArrayMin[0];
                std::cout << "\n";

                std::cout << "The sum value is: " << nieuwArraySum[0];
                std::cout << "\n";

                std::cout << "The prod value is: " << nieuwArrayProd[0];
                std::cout << "\n";

                std::cout << ms << "ms elapsed";
                std::cout << "\n";
                std::cout << "\n";


                std::cout << "-----------------------------ASYNCHROON-----------------------" << "\n";*/
                
                cudaStream_t stream1, stream2, stream3, stream4 ;
                cudaStreamCreate (&stream1);
                cudaStreamCreate (&stream2);
                cudaStreamCreate (&stream3);
                cudaStreamCreate (&stream4);

                cudaEvent_t start_cuda_RedA, stop_cuda_RedA;
                cudaEventCreate(&start_cuda_RedA);
                cudaEventCreate(&stop_cuda_RedA);
                cudaEventRecord(start_cuda_RedA);

                // MAX
                int *inputArrayGPURedMaxA;
                cudaMalloc((void**)&inputArrayGPURedMaxA, sizeof(int) * size);
                cudaMemcpyAsync(inputArrayGPURedMaxA, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice, stream1);

                maxDetectionGPUReduction<<<gridSize, blockSize, 0, stream1>>>(inputArrayGPURedMaxA, size);
                // MIN
                int *inputArrayGPURedMinA;
                cudaMalloc((void**)&inputArrayGPURedMinA, sizeof(int) * size);
                cudaMemcpyAsync(inputArrayGPURedMinA, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice, stream2);

                minDetectionGPUReduction<<<gridSize, blockSize, 0, stream2>>>(inputArrayGPURedMinA, size);
                // SUM 
                int *inputArrayGPURedSumA;
                cudaMalloc((void**)&inputArrayGPURedSumA, sizeof(int) * size);
                cudaMemcpyAsync(inputArrayGPURedSumA, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice, stream3);

                sumDetectionGPUReduction<<<gridSize, blockSize, 0, stream3>>>(inputArrayGPURedSumA, size);
                //Prod
                int *inputArrayGPURedProdA;
                cudaMalloc((void**)&inputArrayGPURedProdA, sizeof(int) * size);
                cudaMemcpyAsync(inputArrayGPURedProdA, inputArray.data(), sizeof(int) * size, cudaMemcpyHostToDevice, stream4);

                prodDetectionGPUReduction<<<gridSize, blockSize, 0, stream4>>>(inputArrayGPURedProdA, size);


                cudaStreamSynchronize(stream1);
                std::vector<int> nieuwArrayMaxA(size);

                cudaMemcpyAsync(nieuwArrayMaxA.data(), inputArrayGPURedMaxA, sizeof(int) * size, cudaMemcpyDeviceToHost, stream1);

                cudaStreamSynchronize(stream2);
                std::vector<int> nieuwArrayMinA(size);

                cudaMemcpyAsync(nieuwArrayMinA.data(), inputArrayGPURedMinA, sizeof(int) * size, cudaMemcpyDeviceToHost, stream2);

                cudaStreamSynchronize(stream3);
                std::vector<int> nieuwArraySumA(size);

                cudaMemcpyAsync(nieuwArraySumA.data(), inputArrayGPURedSumA, sizeof(int) * size, cudaMemcpyDeviceToHost, stream3);

                cudaStreamSynchronize(stream4);
                std::vector<int> nieuwArrayProdA(size);

                cudaMemcpyAsync(nieuwArrayProdA.data(), inputArrayGPURedProdA, sizeof(int) * size, cudaMemcpyDeviceToHost, stream4);

                cudaEventRecord(stop_cuda_RedA);
                cudaEventSynchronize(stop_cuda_RedA);
                ms = 0;
                cudaEventElapsedTime(&ms, start_cuda_RedA, stop_cuda_RedA);
                time_asynchroon += ms;

                /*std::cout << "The max value is: " << nieuwArrayMaxA[0];
                std::cout << "\n";

                std::cout << "The min value is: " << nieuwArrayMinA[0];
                std::cout << "\n";

                std::cout << "The sum value is: " << nieuwArraySumA[0];
                std::cout << "\n";

                std::cout << "The prod value is: " << nieuwArrayProdA[0];
                std::cout << "\n";*/

                cudaFree(inputArrayGPURedMaxA);
                cudaFree(inputArrayGPURedMinA);
                cudaFree(inputArrayGPURedSumA);
                cudaFree(inputArrayGPURedProdA);

                /*std::cout << ms << "ms elapsed";
                std::cout << "\n";
                std::cout << "\n";*/
            }
            writeToCSV(threads, ((size + threads- 1) / threads), size, time_synchroon/gem, false);
            writeToCSV(threads, ((size + threads- 1) / threads), size, time_asynchroon/gem, true);

            std::cout << "Threads: " << threads << " blocks: " << ((size + threads- 1) / threads) << "\n";
            std::cout << "Time synchroon: " << time_synchroon/gem << "\n";
            std::cout << "Time asynchroon: " << time_asynchroon/gem << "\n";
            std::cout << "\n";
        }
    }

    return 0;
}