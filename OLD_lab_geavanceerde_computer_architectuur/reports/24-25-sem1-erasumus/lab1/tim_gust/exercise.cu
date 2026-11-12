#include <iostream>
#include <cuda_runtime.h>

__global__ void flipArray(int *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size / 2) {
        int temp = array[idx];
        array[idx] = array[size - idx - 1];
        array[size - idx - 1] = temp;
    }
}

int main(int argc, char **argv) {

    if(argc!= 2){
        std::cerr<<"Verkeerd aantal argumenten"<< std::endl;
        return -1;
    }

    const int size = std::atoi(argv[1]);
    int inputArray[size] = {};
    for(int i = 0; i < size; i++){
        inputArray[i] = i;
    }
    for (int i = 0; i < size; ++i) {
        std::cout << inputArray[i] << ", ";
    }
    std::cout << "\n";

    int *outputArray;
    cudaMalloc((void**)&outputArray, sizeof(int) * size);//allocate room for the outputArray on the GPU
    cudaMemcpy(outputArray, inputArray, sizeof(int) * size, cudaMemcpyHostToDevice); //copy the input into the destination

    //dim3 blockSize(256); 
    //dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    dim3 blockSize(size);
    dim3 gridSize(100);

    flipArray<<<gridSize, blockSize>>>(outputArray, size);
    cudaMemcpy(inputArray, outputArray, sizeof(int) * size, cudaMemcpyDeviceToHost);//copy the output into the input

    for (int i = 0; i < size; ++i) {
        std::cout << inputArray[i] << ", ";
    }
    std::cout << "\n";

    cudaFree(outputArray);//deallocate memory on the GPU

    return 0;
}