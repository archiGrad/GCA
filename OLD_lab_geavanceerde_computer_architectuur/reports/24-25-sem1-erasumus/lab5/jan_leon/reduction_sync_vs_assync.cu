#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>


#define ARRAY_SIZE 2222
#define THREADS_PER_BLOCK 256

__global__ void reductionKernel(int* input, int* result, int size, int operation) {
    __shared__ int sharedData[THREADS_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialisatie in shared memory
    if (idx < size) {
        sharedData[threadIdx.x] = input[idx];
    } else {
        if (operation == 2)      // Voor minimum
            sharedData[threadIdx.x] = INT_MAX;
        else if (operation == 3) // Voor maximum
            sharedData[threadIdx.x] = INT_MIN;
        else if (operation == 1) // Voor product
            sharedData[threadIdx.x] = 1;
        else                     // Voor som
            sharedData[threadIdx.x] = 0;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (operation == 0)       // Som
                sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
            else if (operation == 1)  // Product
                sharedData[threadIdx.x] *= sharedData[threadIdx.x + stride];
            else if (operation == 2)  // Minimum
                sharedData[threadIdx.x] = min(sharedData[threadIdx.x], sharedData[threadIdx.x + stride]);
            else if (operation == 3)  // Maximum
                sharedData[threadIdx.x] = max(sharedData[threadIdx.x], sharedData[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Resultaat opslaan
    if (threadIdx.x == 0) {
        if (operation == 0 || operation == 1) // Som en product accumuleren
            atomicAdd(result, sharedData[0]);
        else if (operation == 2)              // Minimum
            atomicMin(result, sharedData[0]);
        else if (operation == 3)              // Maximum
            atomicMax(result, sharedData[0]);
    }
}


// CPU Function: Generate data
std::vector<int> generate_data(int size) {
    std::vector<int> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = i +1; // Random numbers between 1 and 10
    }
    return data;
}


// Synchronous Execution
double synchronous_execution(int array_count, int size, std::vector<int> &results) {
    auto start_sync = std::chrono::high_resolution_clock::now();

    for (int op = 0; op < array_count; ++op) {
        auto data = generate_data(size);
        int *d_input, *d_output;
        int result = (op%4 == 2) ? INT_MAX : 0; // Neutral for multiplication or summation

        cudaMalloc(&d_input, size * sizeof(int));
        cudaMalloc(&d_output, sizeof(int));
        cudaMemcpy(d_output, &result, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input, data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

        // Kernel launch
        reductionKernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_input, d_output, size, op%4);

        cudaDeviceSynchronize();
        cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

        results.push_back(result); // Store result for this kernel
        cudaFree(d_input);
        cudaFree(d_output);
    }

    auto end_sync = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_sync - start_sync).count();
}


void executeAsync(int* h_input, int size) {
    int* d_input;
    int* d_result;
    int h_result;

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (int op = 0; op < 4; ++op) {
        h_result = (op == 2) ? INT_MAX : 0;
        cudaMemcpyAsync(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice, streams[op]);

        reductionKernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[op]>>>(d_input, d_result, size, op);
        cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, streams[op]);

        cudaStreamSynchronize(streams[op]);
        std::cout << "Async result (operation " << op << "): " << h_result << std::endl;
    }

    for (int i = 0; i < 4; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_input);
    cudaFree(d_result);
}

// Asynchronous Execution
double asynchronous_execution(int array_count, int size, std::vector<int> &results) {
    auto start_async = std::chrono::high_resolution_clock::now();
    std::vector<int*> d_inputs(array_count), d_outputs(array_count);
    std::vector<cudaStream_t> streams(array_count);

    // Create CUDA streams
    for (int op = 0; op < array_count; ++op) {
        cudaStreamCreate(&streams[op]);
        cudaMalloc(&d_inputs[op], size * sizeof(int));
        cudaMalloc(&d_outputs[op], sizeof(int));
    }

    for (int op = 0; op < array_count; ++op) {
        auto data = generate_data(size);
        int result = (op%4 == 2) ? INT_MAX : 0; // Neutral for multiplication or summation

        // Asynchronous data transfer
        cudaMemcpyAsync(d_inputs[op], data.data(), size * sizeof(int), cudaMemcpyHostToDevice, streams[op]);
        cudaMemcpyAsync(d_outputs[op], &result, sizeof(int), cudaMemcpyHostToDevice, streams[op]);

        // Kernel launch
        reductionKernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[op]>>>(d_inputs[op], d_outputs[op], size, op%4);

        cudaMemcpyAsync(&result, d_outputs[op], sizeof(int), cudaMemcpyDeviceToHost, streams[op]);
        cudaStreamSynchronize(streams[op]);

        results.push_back(result); // Store result for this kernel
    }

    // Cleanup
    for (int op = 0; op < array_count; ++op) {
        cudaFree(d_inputs[op]);
        cudaFree(d_outputs[op]);
        cudaStreamDestroy(streams[op]);
    }

    auto end_async = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_async - start_async).count();
}

// Main Function
int main() {
    int aantalIteratiesVan4 = 1;
    int array_count = 4; // 4 operations: sum, prod, min, max

    int initial_size = 1;  // Starting size
    int num_elements = 25;  // Number of elements you want //beter niet hoger gaan dan 25 want 2^25 = 33554432	is meeeeeer dan genoeg

    std::vector<int> sizes;
    sizes.push_back(initial_size);
    sizes.push_back(initial_size); // eerste 2 keer doen want eerste keer doet er altijd veel langer over dan normaal

    // intializeer de verschillende groottes van de input-array
    for (int i = 1; i < num_elements; i++) {
        sizes.push_back(sizes[i] *2);  // add previous value by 2
    }

    // Open CSV file
    std::ofstream csv_file("reduction_results.csv");
    csv_file << "Array_Size,Execution_Type,Operation,Time_ms\n";
    std::cout << "sizes.size() = " << sizes.size() << std::endl;
    for (int size : sizes) {
        std::cout << size << std::endl;
    }

    for (int size : sizes) {
        double sync_total = 0.0, async_total = 0.0;

        std::cout << "Array Size: " << size << "\n";

        for (int i = 0; i < 10; ++i) {
            std::vector<int> sync_results, async_results;

            auto sync_start = std::chrono::high_resolution_clock::now();
            sync_total += synchronous_execution(array_count * aantalIteratiesVan4, size, sync_results);
            auto sync_end = std::chrono::high_resolution_clock::now();

            auto async_start = std::chrono::high_resolution_clock::now();
            async_total += asynchronous_execution(array_count * aantalIteratiesVan4, size, async_results);
            auto async_end = std::chrono::high_resolution_clock::now();

            // Calculate elapsed times
            double sync_time = std::chrono::duration_cast<std::chrono::milliseconds>(sync_end - sync_start).count();
            double async_time = std::chrono::duration_cast<std::chrono::milliseconds>(async_end - async_start).count();

            // // Log to console
            // std::cout << "Iteration " << i + 1 << " Results:\n";
            // std::cout << "  Synchronous Results: ";
            // for (int res : sync_results) std::cout << res << " ";
            // std::cout << "\n";

            // std::cout << "  Asynchronous Results: ";
            // for (int res : async_results) std::cout << res << " ";
            // std::cout << "\n";

            // // Write synchronous results to CSV
            // for (int op = 0; op < array_count; ++op) {
            //     csv_file << size << "," << i + 1 << ",Synchronous,Operation_" << op << "," << sync_time << "\n";
            // }

            // // Write asynchronous results to CSV
            // for (int op = 0; op < array_count; ++op) {
            //     csv_file << size << "," << i + 1 << ",Asynchronous,Operation_" << op << "," << async_time << "\n";
            // }
        }

        double sync_avg = sync_total / 10.0;
        double async_avg = async_total / 10.0;

        // Write synchronous results to CSV
        for (int op = 0; op < array_count; ++op) {
            csv_file << size << ",Synchronous,Operation_" << op << "," << sync_avg << "\n";
        }
        // Write asynchronous results to CSV
        for (int op = 0; op < array_count; ++op) {
            csv_file << size << ",Asynchronous,Operation_" << op << "," << async_avg << "\n";
        }

        // std::cout << "Average Synchronous Execution Time: " << sync_avg << " ms\n";
        // std::cout << "Average Asynchronous Execution Time: " << async_avg << " ms\n";
        // std::cout << "-----------------------------------\n";
    }

    // Close CSV file
    csv_file.close();

    return 0;
}

