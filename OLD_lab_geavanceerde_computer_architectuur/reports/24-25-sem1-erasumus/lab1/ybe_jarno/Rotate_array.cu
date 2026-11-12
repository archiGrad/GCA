#include <cstdio>
#include <iostream>

#include <stdio.h>
#include <cuda.h>

__global__ void flipArray(int *d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / 2) {
        // Calculate the corresponding index from the other end
        int temp = d_array[idx];
        d_array[idx] = d_array[size - 1 - idx];
        d_array[size - 1 - idx] = temp;
    }
}

void flip(int *h_array, int size, int blocks, int threads) {
    int *d_array;
    size_t bytes = size * sizeof(int);
    cudaMalloc((void **)&d_array, bytes);
    
    // Copy the host array to device
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    clock_t begin = clock();
    flipArray<<<blocks, threads>>>(d_array, size);
    clock_t end = clock();
    double time = (double) (end-begin) / CLOCKS_PER_SEC;
    printf("%f", time);
    
    // Copy the result back to host
    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

int main() {
    int h_array[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int size = sizeof(h_array) / sizeof(h_array[0]);

    printf("Original array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    printf("Flip with 1 block and 9 threads time: ");
    flip(h_array, size, 1, 9);
//    for (int i = 0; i < size; i++) {
//        printf("%d ", h_array[i]);
//    }
    printf("\n");


    printf("Flip with 3 block and 3 threads time: ");
    flip(h_array, size, 3, 3);
//    for (int i = 0; i < size; i++) {
//        printf("%d ", h_array[i]);
//    }
   printf("\n");


    printf("Flip with 9 block and 1 threads time: ");
    flip(h_array, size, 9, 1);
//    for (int i = 0; i < size; i++) {
//        printf("%d ", h_array[i]);
//    }
    printf("\n");   
    
    return 0;
}
