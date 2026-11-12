#include "stdio.h"
#include <chrono>

void flip_array_cpu(int *in, int *out, int n){
  for (int i=0; i<n; i++){
    int idx = n-i-1;
    out[idx] = in[i];
  }
}

__global__ void flip_array(int *in, int *out, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = threadIdx.x;
  int idx_out = n-idx-1;
  //printf("thread idx: %d - idxout: %d \n", i, idx_out);
  out[idx_out] = in[idx];
}

int main() {
  // define array size
  const int size = 10000;

  // declare array
  int my_array[size];
  for(int i=0; i<size; i++){
    my_array[i] = i;
  };

 /*  // print array
  printf("input array: \n");
  for (int i=0; i<size; i++){
    printf("%d, ", my_array[i]);
  };
  printf("\n"); */


  // declare output array
  int out_array[size];

  // flip array on cpu
  const auto start = std :: chrono :: steady_clock :: now () ;
  flip_array_cpu(my_array, out_array, size);
  const auto end = std :: chrono :: steady_clock :: now () ;
  const std :: chrono :: duration <double> elapsed_seconds{end - start};
  printf("ellapsed cpu time: %f s", elapsed_seconds);
  printf("\n");


  /* // print array
  printf("output array: \n");
  for (int i=0; i<size; i++){
    printf("%d, ", out_array[i]);
  };
  printf("\n"); */


  // do the same on the GPU
  printf("GPU: \n");
  int malloc_size = size * sizeof(int);

  // allocate memory on gpu
  int* out_gpu;
  cudaMalloc(&out_gpu, malloc_size);
  int* in_gpu;
  cudaMalloc(&in_gpu, malloc_size);

  // copy arrays from host to device (cpu=>gpu)
  cudaMemcpy(in_gpu, my_array, malloc_size, cudaMemcpyHostToDevice);

  // run kernel
  int nr_blocks = 2;
  int nr_threads = size/2;

  cudaEvent_t start_cuda, stop_cuda;
  cudaEventCreate(&start_cuda);
  cudaEventCreate(&stop_cuda);
  cudaEventRecord(start_cuda);
  flip_array<<<nr_blocks, nr_threads>>>(in_gpu, out_gpu, size); 
  
  //cudaDeviceSynchronize();
  cudaEventRecord(stop_cuda);
  cudaEventSynchronize(stop_cuda);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda);
  printf("gpu time %f ms", milliseconds);

  // copy result from device to host (gpu=>cput)
  int out_cpu[size];
  cudaMemcpy(out_cpu, out_gpu, malloc_size, cudaMemcpyDeviceToHost);

  /* // print array
  printf("output array: \n");
  for (int i=0; i<size; i++){
    printf("%d, ", out_cpu[i]);
  };
  printf("\n"); */

}