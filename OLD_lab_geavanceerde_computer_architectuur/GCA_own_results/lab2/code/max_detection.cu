#include "stdio.h"
#include <chrono>

void print_array(int *array, int size){
  printf("[");
  for(int i=0; i<size; i++){
    printf("%d, ", array[i]);
  }
  printf("] \n");
}

int max_cpu(int *array, int size){
  int max_val = 0;
  for(int i=0; i<size; i++){
    if (array[i] > max_val){
      max_val = array[i];
    }
  }
  return max_val;
}

__global__ void max_gpu_atomic(int *array, int size, int *res){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  atomicMax(res, array[idx]);
}

__global__ void max_gpu_reduce(int *array, int size, int stride, int offset){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // store max at correct idx
  array[stride*idx] = max(array[stride*idx], array[stride*idx+offset]);

}

__global__ void maxReduceSharedMem(int* array, int size){
  extern __shared__ int sdata[]; //shared memory

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // load elements into shared memory
  if (idx < size){
    sdata[tid] = array[idx];
  }
  else{
    sdata[tid] = INT_MIN;
  }
  __syncthreads();


  // perform reduction
  for(int s = 1; s<blockDim.x; s*=2){
    if (threadIdx.x % (2*s) == 0){
      sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    }
    __syncthreads();
  }

  // thread zero writes results to main memory
  if (threadIdx.x == 0){
    array[0] = sdata[0];
  }
}



__global__ void maxReduceGlobalMem(int* array, int size){

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // perform reduction
  for(int s = 1; s<blockDim.x; s*=2){
    if (threadIdx.x % (2*s) == 0){
      array[threadIdx.x] = max(array[threadIdx.x], array[threadIdx.x + s]);
    }
    __syncthreads();
  }

}

int main() {
  // create array
  const int size = 1024;//1024;
  int array[size];
  for( int i=0; i<size; i++){
    array[i] = rand();
  }
  printf("input array: ");
  print_array(array, size);

  /*----------------------- CPU loop ---------------------------------*/ 
  const auto start = std::chrono::steady_clock::now();
  int max_val = max_cpu(array, size);
  const auto end = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed_seconds{end-start};
  printf("max value: %d \n", max_val);
  printf("time  CPU: %f ms\n", elapsed_seconds*1000);

  /*----------------------- GPU atomicoperations ---------------------------------*/ 
  // init int on host(cpu)
  int res = 0;
  // alocate memory on gpu (device)
  int* array_gpu;
  cudaMalloc(&array_gpu, size*sizeof(int));
  cudaMemcpy(array_gpu, array, size*sizeof(int), cudaMemcpyHostToDevice);


  int* max_val_gpu;
  cudaMalloc(&max_val_gpu, sizeof(int));
  cudaMemcpy(max_val_gpu, &res, sizeof(int), cudaMemcpyHostToDevice);
  

  // run kernel
  int nr_blocks = 1;
  int nr_threads = size;
  cudaEvent_t start_cuda, stop_cuda;
  cudaEventCreate(&start_cuda);
  cudaEventCreate(&stop_cuda);
  cudaEventRecord(start_cuda);
  max_gpu_atomic<<<nr_blocks, nr_threads>>>(array_gpu, size, max_val_gpu);
  cudaDeviceSynchronize();
  cudaEventRecord(stop_cuda);
  cudaEventSynchronize(stop_cuda);
  float ms;
  cudaEventElapsedTime(&ms, start_cuda, stop_cuda);
  printf("GPU atomic: %f ms \n", ms);

  // copy maxval back to cpu
  cudaMemcpy(&res, max_val_gpu, sizeof(int), cudaMemcpyDeviceToHost);

  // freememory
  cudaFree(array_gpu);
  cudaFree(max_val_gpu);

  printf("GPU atomic max val: %d \n", res);


  /*----------------------- GPU reduction with CPU synchronization ---------------------------------*/ 
  // CPU synchronization cause kernel call are made consecutively
  // all treads start and end at same point (for one kernel call)

  // alocate memory on GPU
  int* deviceArray;
  cudaMalloc(&deviceArray, size*sizeof(int));
  cudaMemcpy(deviceArray, array, size*sizeof(int), cudaMemcpyHostToDevice);

  // nr of reductions
  int nr_red = log2(size);
  //printf("nr reductions: %d \n", nr_red);

  cudaEvent_t start_cuda2, stop_cuda2;
  cudaEventCreate(&start_cuda2);
  cudaEventCreate(&stop_cuda2);
  cudaEventRecord(start_cuda2);

  for(int i=0; i<nr_red; i++){
    // invoke kernel to do reduction
    int nr_threads = size/2/(i+1);
    int nr_blocks = 1;
    int stride = pow(2, i+1);
    int offset = stride/2;
    //printf("reduction level: %d - stride: %d - offset: %d \n", i, stride, offset);
    max_gpu_reduce<<<nr_threads, nr_blocks>>>(deviceArray, size, stride, offset);
  }

  cudaDeviceSynchronize();
  cudaEventRecord(stop_cuda2);
  cudaEventSynchronize(stop_cuda2);
  float ms2;
  cudaEventElapsedTime(&ms2, start_cuda2, stop_cuda2);
  printf("GPU reduction CPU sync: %f ms \n", ms2);

  // copy maxval back to cpu
  int resArray[size];
  cudaMemcpy(resArray, deviceArray, sizeof(int), cudaMemcpyDeviceToHost);
  //print_array(resArray, size);
  printf(" cpu reduction res: %d \n", resArray[0]);

  /*----------------------- GPU reduction with block synchronization and shared memory ---------------------------------*/ 
  // alocate memory on GPU
  int* deviceArray2;
  cudaMalloc(&deviceArray2, size*sizeof(int));
  cudaMemcpy(deviceArray2, array, size*sizeof(int), cudaMemcpyHostToDevice);

  int n = size;
  int numBlocks = 1; // if you want to use more blocks the code should be adjusted by adding a while loop and launching the kernel multiple times
  int block_size = size/numBlocks;

  cudaEvent_t start_cuda3, stop_cuda3;
  cudaEventCreate(&start_cuda3);
  cudaEventCreate(&stop_cuda3);
  cudaEventRecord(start_cuda3);

  // launch the kernel (assume enough threads and blocks to do all reductions in one time if not need to call the kernel again)
  maxReduceSharedMem<<<numBlocks, block_size, block_size*sizeof(int)>>>(deviceArray2, size);

  cudaDeviceSynchronize();
  cudaEventRecord(stop_cuda3);
  cudaEventSynchronize(stop_cuda3);
  float ms3;
  cudaEventElapsedTime(&ms3, start_cuda3, stop_cuda3);
  printf("GPU reduction shared mem: %f ms \n", ms3);

  // copy maxval back to cpu
  int resArray2[size];
  cudaMemcpy(resArray2, deviceArray2, sizeof(int), cudaMemcpyDeviceToHost);
  printf("GPU shared memory res: %d \n", resArray2[0]);
  //print_array(resArray2, size);

    /*----------------------- GPU reduction with block synchronization and host memory ---------------------------------*/ 

// alocate memory on GPU
  int* deviceArray3;
  cudaMalloc(&deviceArray3, size*sizeof(int));
  cudaMemcpy(deviceArray3, array, size*sizeof(int), cudaMemcpyHostToDevice);

  
  cudaEvent_t start_cuda4, stop_cuda4;
  cudaEventCreate(&start_cuda4);
  cudaEventCreate(&stop_cuda4);
  cudaEventRecord(start_cuda4);

  // launch the kernel (assume enough threads and blocks to do all reductions in one time if not need to call the kernel again)
  maxReduceGlobalMem<<<numBlocks, block_size, block_size*sizeof(int)>>>(deviceArray2, size);

  cudaDeviceSynchronize();
  cudaEventRecord(stop_cuda4);
  cudaEventSynchronize(stop_cuda4);
  float ms4;
  cudaEventElapsedTime(&ms4, start_cuda4, stop_cuda4);
  printf("GPU reduction global mem: %f ms \n", ms4);

  // copy maxval back to cpu
  int resArray3[size];
  cudaMemcpy(resArray3, deviceArray3, sizeof(int), cudaMemcpyDeviceToHost);
  printf("GPU global memory res: %d \n", resArray2[0]);





}