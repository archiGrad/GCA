#include <vector>
#include <device_launch_parameters.h>
#include <cstdint>      
#include <iostream>     
#include <cuda_runtime.h>
#include <iomanip>      // For std::setw and pretty printing

#define M 960       
#define N 1280      
#define C 3         
#define OFFSET 16   

uint8_t* get_image_array(void){
    /*
     * Get the data of an (RGB) image as a 1D array.
     * 
     * Returns: Flattened image array.
     * 
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the first row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     * 
     */        
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./anna.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    
    // Read the image
    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
    
    // Close the file
    fclose(imageFile);
        
    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}


void save_image_array(uint8_t* image_array){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./output_image_part2.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Configure the file
    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
    
    // Write the image
    fwrite(image_array, 1, M*N*C, imageFile);
    
    // Close the file
    fclose(imageFile);
}


void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void reverseRGBKernel(uint8_t* input, uint8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Determine the color component based on idx
        int colorIdx = idx % C; // 0 for R, 1 for G, 2 for B
        uint8_t colorValue = input[idx];

        if (colorIdx == 0) { // If it's the red color component (R)
            // Check if value is between 100 and 200
            if (colorValue >= 100 && colorValue <= 200) {
                // Apply the transformation (x = (x % 25) * 10)
                output[idx] = (colorValue % 25) * 10;
            } else {
                // Invert R component
                output[idx] = 255 - colorValue;
            }
        } else {
            // Invert G and B components (no special rule)
            output[idx] = 255 - colorValue;
        }
    }
}


// Structure to hold timing results
struct TimingResult {
    int threadsPerBlock;
    int numBlocks;
    float kernelTime;
    float totalTime;
};

// Function to run the kernel with timing
TimingResult runKernelWithTiming(uint8_t* d_input, uint8_t* d_output, int imageSize, int threadsPerBlock) {
    TimingResult result;
    result.threadsPerBlock = threadsPerBlock;
    result.numBlocks = (imageSize + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Launch kernel
    reverseRGBKernel<<<result.numBlocks, threadsPerBlock>>>(d_input, d_output, imageSize);
    
    // End timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&result.kernelTime, start, stop);

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

int main(void) {
    // Read the image
    uint8_t* image_array = get_image_array();
    const int imageSize = M * N * C;
    
    // Allocate device memory
    uint8_t *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, imageSize * sizeof(uint8_t)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, imageSize * sizeof(uint8_t)), "cudaMalloc d_output");
    
    // Copy image data to device
    checkCudaError(cudaMemcpy(d_input, image_array, imageSize * sizeof(uint8_t), 
                             cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Vector to store timing results
    std::vector<TimingResult> timingResults;

    // Test different thread configurations
    // Starting from 32 (1 warp) and doubling up to 1024  // je start toch van 2?
    std::cout << "\nRunning performance tests...\n" << std::endl;
    std::cout << std::setw(15) << "Threads/Block" 
              << std::setw(15) << "Blocks" 
              << std::setw(15) << "Kernel (ms)"
              << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    for (int threads = 2; threads <= 1024; threads += 2) {
        // Run kernel multiple times for each configuration to get average
        const int NUM_ITERATIONS = 10;
        TimingResult avgResult = {threads, 0, 0.0f, 0.0f};
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            TimingResult result = runKernelWithTiming(d_input, d_output, imageSize, threads);
            avgResult.kernelTime += result.kernelTime;
            avgResult.numBlocks = result.numBlocks;
        }
        
        // Calculate averages
        avgResult.kernelTime /= NUM_ITERATIONS;
        
        // Print results
        std::cout << std::setw(15) << avgResult.threadsPerBlock
                  << std::setw(15) << avgResult.numBlocks
                  << std::setw(15) << std::fixed << std::setprecision(3) << avgResult.kernelTime
                  << std::endl;
        
        timingResults.push_back(avgResult);
    }

    // Find the best configuration
    auto bestConfig = std::min_element(timingResults.begin(), timingResults.end(),
        [](const TimingResult& a, const TimingResult& b) {
            return a.kernelTime < b.kernelTime;
        });

    std::cout << "\nBest configuration:" << std::endl;
    std::cout << "Threads per block: " << bestConfig->threadsPerBlock << std::endl;
    std::cout << "Number of blocks: " << bestConfig->numBlocks << std::endl;
    std::cout << "Kernel time: " << bestConfig->kernelTime << " ms" << std::endl;

    // Use the best configuration for the final output
    uint8_t* new_image_array = (uint8_t*)malloc(imageSize * sizeof(uint8_t));
    checkCudaError(cudaMemcpy(new_image_array, d_output, imageSize * sizeof(uint8_t), 
                             cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Save the image
    save_image_array(new_image_array);

    // Clean up
    free(image_array - OFFSET);
    free(new_image_array);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}