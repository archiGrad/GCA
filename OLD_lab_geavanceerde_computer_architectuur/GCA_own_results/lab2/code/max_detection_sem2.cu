#include "stdio.h"
#include <chrono>
#include <iostream>
using namespace std;

__global__ void max_atomic(int arr[], int size){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    atomicMax(arr, arr[idx]);
}

__global__ void max_reduce(int arr[], int size){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int reduction_steps = log2f(size);
    for(int i=0; i<reduction_steps ;i++){
        int idx_start = int(pow(float(2), float(i+1))) * idx;
        int offset = int(pow(float(2), float(i)));
        if (idx_start+offset<size){
            if (arr[idx_start] < arr[idx_start+offset]){
                arr[idx_start] = arr[idx_start+offset];
            }    
        }
        __syncthreads();
    }
}

int max_cpu(int arr[], int size){
    int maxval = 0;
    for (int i=0; i<size; i++){
        if (arr[i] > maxval){
            maxval = arr[i];
        }
    }
    return maxval;
}

int main(){
    // generate array
    int size = 2048;
    int* array = new int[size];
    for (int i=0; i<size; i++){
        array[i] = rand() % 10000;
        cout << array[i] << " ";
    }
    cout << endl;

    // max val on cpu
    const auto start = std::chrono::steady_clock::now();
    int max_val_cpu =  max_cpu(array, size);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds{end-start};
    cout << "max value: " << max_val_cpu << " time cpu: " << elapsed_seconds.count() * 1000 << " ms\n";
    cout << endl;

    // max val on GPU (atomic)
    // memory allocation on gpu
    int* gpu_array;
    cudaMalloc(&gpu_array, size*sizeof(int));
    // memory migration to gpu
    cudaMemcpy(gpu_array, array, size*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda);

    // call kernel
    max_atomic<<<1, size>>>(gpu_array, size);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    float ms;
    cudaEventElapsedTime(&ms, start_cuda, stop_cuda);

    // copy result back to cpu
    int* res_arr = new int[size];
    cudaMemcpy(res_arr, gpu_array, size*sizeof(int), cudaMemcpyDeviceToHost);

    cout << "max val GPU atomic: " << res_arr[0] << " time: " << ms << "ms";
    cout << endl;

    // max val on GPU with reduction 
    // memory allocation on gpu
    int* gpu_array2;
    cudaMalloc(&gpu_array2, size*sizeof(int));
    // memory migration to gpu
    cudaMemcpy(gpu_array2, array, size*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_cuda2, stop_cuda2;
    cudaEventCreate(&start_cuda2);
    cudaEventCreate(&stop_cuda2);
    cudaEventRecord(start_cuda2);

    // call kernel
    max_reduce<<<1, size/2>>>(gpu_array2, size);

    cudaDeviceSynchronize();
    cudaEventRecord(stop_cuda2);
    cudaEventSynchronize(stop_cuda2);
    float ms2;
    cudaEventElapsedTime(&ms2, start_cuda2, stop_cuda2);

    // copy result back to cpu
    int* res_arr2 = new int[size];
    cudaMemcpy(res_arr2, gpu_array2, size*sizeof(int), cudaMemcpyDeviceToHost);

    cout << "max val GPU reduce: " << res_arr2[0] << " time: " << ms2 << "ms";
    cout << endl;










}