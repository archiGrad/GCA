#include <iostream>
#include <chrono>
#include <climits>
using namespace std;
using namespace std::chrono;

__global__ void getMaxGPU(int *array_in, int *block_max, int size) {
    extern __shared__ int shared_max[];  // Shared memory for block reduction

    int tid = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    shared_max[tid] = (global_index < size) ? array_in[global_index] : INT_MIN;
    __syncthreads();

    // Parallel reduction within each block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && global_index + stride < size) {
            // if (shared_max[tid] < shared_max[tid + stride]){
            //     shared_max[tid] = shared_max[tid + stride];
            // }
            shared_max[tid] =  max(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // Store the block result
    if (tid == 0) {
        block_max[blockIdx.x] = shared_max[0];
    }
}

__global__ void getMaxAtomic(int* input, int* maxVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        *maxVal = INT_MIN;  // Initialize the maxVal to INT_MIN
    }
    __syncthreads();  // Make sure initialization is done
    
    // Check if the index is within bounds
    if (idx < n) {
        atomicMax(maxVal, input[idx]);
    }
}

int getMaxCPU(int *array_in, int size) {
    int max = INT_MIN;
    for (int i = 0; i < size; i++) {
        if (array_in[i] > max) {
            max = array_in[i];
        }
    }
    return max;
}

int main() {
    const int size = (1000000 );  // Larger array for performance measurement

    int* h_in = new int[size];

    // Fill the array
    for (int i = 0; i < size; i++) {
        h_in[i] = i;
    }

    // Start timing
    auto start_cpu = high_resolution_clock::now();

    int max = getMaxCPU(h_in, size);

    // Stop timing
    auto stop_cpu = high_resolution_clock::now();
    auto duration_cpu = duration_cast<microseconds>(stop_cpu - start_cpu);
    cout << "Max value: " << max << endl;
    cout << "CPU duration: " << duration_cpu.count() << " microseconds" << endl;

    // -----------------------------------------------------------------------------------------
    // GPU implementation using reduction
    // -----------------------------------------------------------------------------------------
    int *d_in, *d_out;

   // Allocate memory on the GPU
    cudaMalloc(&d_in, size * sizeof(int));
    cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int current_size = size;
    // + threadsPerBlock - 1 zorgt ervoor dat er naar boven wordt afgerond
    int blocks = (current_size + threadsPerBlock - 1) / threadsPerBlock;

    // Temporary memory for the reduction
    cudaMalloc(&d_out, blocks * sizeof(int));

    // Tijdopname starten
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (current_size > 1) {
        // Launch kernel with current size
        getMaxGPU<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_in, d_out, current_size);

        // Update sizes and pointers for the next reduction step
        current_size = blocks;
        blocks = (current_size + threadsPerBlock - 1) / threadsPerBlock;

        // Swap d_in and d_out for the next pass
        int *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

   
    int max_gpu = INT_MIN;
    cudaMemcpy(&max_gpu, d_in, sizeof(int), cudaMemcpyDeviceToHost);

    // Tijdopname stoppen
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration_gpu = 0;
    cudaEventElapsedTime(&duration_gpu, start, stop);

    cout << "Max value using gpu: " << max_gpu << endl;
    cout << "GPU duration: " << duration_gpu*1000 << " microseconds" << endl;

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

   // -----------------------------------------------------------------------------------------
    // GPU implementation using atomics
    // -----------------------------------------------------------------------------------------
    int *d_in2, *d_max;  // Rename d_out to d_max for clarity

    // Allocate memory on the GPU
    cudaMalloc(&d_in2, size * sizeof(int));
    cudaMalloc(&d_max, sizeof(int));  // Only need space for one integer
    cudaMemcpy(d_in2, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock2 = 256;
    int blocks2 = (size + threadsPerBlock2 - 1) / (threadsPerBlock2);

    // Start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    getMaxAtomic<<<blocks2, threadsPerBlock2>>>(d_in2, d_max, size);

    cudaEventRecord(stop);

    // Copy result back to CPU
    int max_atomic;
    cudaMemcpy(&max_atomic, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    cout << "Max value using gpu with atomic: " << max_atomic << endl;
    cout << "GPU duration with atomic: " << gpuDuration*1000 << " microseconds" << endl;

    // Clean up
    cudaFree(d_in2);
    cudaFree(d_max);

    return 0;
}
