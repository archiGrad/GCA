#include <cstdio>
#include <iostream>
#include <cmath>

#include <stdio.h>
#include <cuda.h>


__global__ void vectorMaxAtomic (int *A,int *highNumAtomic,int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < numElements) {
        atomicMax(highNumAtomic,A[i]);
     }
}

__global__ void vectorMaxReduction (int *A,int *output,int numElements) {
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<numElements){
        shared_data[tid] = A[idx];
    } else{
        shared_data[tid] = 0;
    }
    __syncthreads();

    for(int step = blockDim.x/2; step >0; step /=2){
        if(tid < step){
            shared_data[tid] = max(shared_data[tid], shared_data[tid+step]);
        }
    }
    __syncthreads();
    if(tid == 0){
        output[blockIdx.x] = shared_data[0];
    }
}





int maxAtomic(int* h_array,int size,int blocks, int threads,int max){
    int *d_array;
    int *d_max;
    cudaMalloc((void **)&d_array, size * sizeof(int));
    cudaMalloc((void **)&d_max, sizeof(int));

    
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &max,sizeof(int),cudaMemcpyHostToDevice);    
    clock_t begin = clock();
    vectorMaxAtomic<<<blocks, threads>>>(d_array,d_max,size);
    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("GPU TIJD; %f  ;", time);
    cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    
    return max;
}

int maxCpu(int* h_array,int j){
    int max = 0;
    clock_t begin = clock();
    for(int i =0;i<j;i++){
        if(h_array[i]>max){
            max = h_array[i];
        }
    }
    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("CPU TIJD; %f ;", time);
    return max;
}

int maxReduction(int* h_array,int size){
    int *d_input, *d_output;
    int threads = 9;
    int blocks = (size + threads - 1) / threads;
    cudaMalloc((void **)&d_input, size * sizeof(int));
    cudaMalloc((void **)&d_output, blocks * sizeof(int));
    
    cudaMemcpy(d_input, h_array, size * sizeof(int), cudaMemcpyHostToDevice); 
    clock_t begin = clock();
    vectorMaxReduction<<<blocks, threads, threads*sizeof(int)>>>(d_input,d_output,size);
    int *h_output = new int[blocks];
    cudaMemcpy(h_output, d_output,blocks*sizeof(int),cudaMemcpyDeviceToHost);
    int max_val = h_output[0];
    for(int i = 1; i<blocks;i++){
        max_val = max(max_val, h_output[i]);
    }


    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("GPU reduction TIJD; %f ; ", time);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;
    
    return max_val;
}

int main() {
    for (int j=10; j<1000;){
        int h_array[j];
        for (int i=0; i<j; i++){
            h_array[i]= rand()%1000;
        }
        int size = sizeof(h_array) / sizeof(h_array[0]);
        int max = 0;
        int threadsPerBlock = 256;

        printf("Array size; %d ;", j);

        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        printf("Blocks per grid; %d ;",blocksPerGrid);
        //max by atomic operation
        int maxA = maxAtomic(h_array,size,blocksPerGrid,threadsPerBlock,max);
        printf("max; %d ;",maxA);
        //max by for-loop on cpu
        int maxC = maxCpu(h_array,j);
        printf("maxCpu;  %d ;",maxC);
        //max by reduction
        int maxR = maxReduction(h_array,size);
        printf("maxReduction; %d ; \n",maxR);
        j += 50;
    }
    return 0;
}
