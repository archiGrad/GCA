#include <vector>
#include <device_launch_parameters.h>
#include <cstdint>      
#include <iostream>     
#include <cuda_runtime.h>
#include <iomanip>      
#include <fstream>       // For file I/O

#define M 9600       
#define N 12800      
#define C 3         
#define OFFSET 16   

uint8_t* get_image_array(void){
    FILE *imageFile;
    imageFile=fopen("./anna.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open input file");
        exit(EXIT_FAILURE);
    }
    
    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
    fclose(imageFile);
    return image_array + OFFSET;
}

void save_image_array(uint8_t* image_array){
    FILE *imageFile;
    imageFile=fopen("./output_image_part2.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    fprintf(imageFile,"P6\n");
    fprintf(imageFile,"%d %d\n", M, N);
    fprintf(imageFile,"255\n");
    fwrite(image_array, 1, M*N*C, imageFile);
    fclose(imageFile);
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


// Coalesced memory access kernel (efficient)
__global__ void coalescedMemoryAccessKernel(uint8_t* input_sorted, uint8_t* output, int pixelCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelCount) {  // <= of < ?
        input_sorted[idx] = 255 - input_sorted[idx];  // Invert red
    }
}

// Un-coalesced memory access kernel (inefficient)
__global__ void uncoalescedMemoryAccessKernel(uint8_t* input, uint8_t* output, int pixelCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixelCount) {
         int redPixel = idx * 3;     // R-kanaal
         input[redPixel] = 255 - redPixel;  // Invert red
    }
}

// Timing structure
struct TimingResult {
    int threadsPerBlock;
    int numBlocks;
    float coalescedTime;
    float uncoalescedTime;
};


void convertRGBtoPlanar(const uint8_t *input, uint8_t *output, int pixelCount) {
    int totalPixels = pixelCount;

    // Pointers naar de juiste posities in de output array
    uint8_t *outputR = output;                // R begint bij het begin
    uint8_t *outputG = output + totalPixels;  // G begint na alle R
    uint8_t *outputB = output + 2 * totalPixels; // B begint na alle G

    for (int i = 0; i < totalPixels; ++i) {
        outputR[i] = input[i * 3];     // R-kanaal
        outputG[i] = input[i * 3 + 1]; // G-kanaal
        outputB[i] = input[i * 3 + 2]; // B-kanaal
    }

    std::cout << "Converted RGB to planar format." << std::endl;
}


// Function to run kernels with timing
TimingResult runKernelsWithTiming(uint8_t* d_input, uint8_t* d_output, uint8_t* d_input_sorted, int pixelCount, int threadsPerBlock) {
    TimingResult result;
    result.threadsPerBlock = threadsPerBlock;

    int numBlocks = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    result.numBlocks = numBlocks;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time coalesced memory access kernel
    cudaEventRecord(start);
    coalescedMemoryAccessKernel<<<numBlocks, threadsPerBlock>>>(d_input_sorted, d_output, pixelCount);
    checkCudaError(cudaGetLastError(), "Kernel launch (Coalesced)");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.coalescedTime, start, stop);

    // Time uncoalesced memory access kernel
    cudaEventRecord(start);
    uncoalescedMemoryAccessKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, pixelCount);
    checkCudaError(cudaGetLastError(), "Kernel launch (Un-coalesced)");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.uncoalescedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

int main(void) {
    uint8_t* image_array = get_image_array();
    uint8_t* image_array_sorted = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    const int imageSize = M * N * C;
    const int pixelCount = M * N;

    // // print image_array
    // std::cout << "Image array: " << std::endl;
    // for (int i = 0; i < 100; i++) {
    //     std::cout << (int)image_array[i] << ", ";
    // }

    convertRGBtoPlanar(image_array, image_array_sorted, pixelCount);

    // // print image_array_sorted
    // std::cout << "Image_array_sorted: " << std::endl;
    // for (int i = 0; i < 100; i++) {
    //     std::cout << (int)image_array_sorted[i] << ", ";
    // }

    uint8_t *d_input, *d_output, *d_input_sorted;
    checkCudaError(cudaMalloc(&d_input, imageSize * sizeof(uint8_t)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, imageSize * sizeof(uint8_t)), "cudaMalloc d_output");
    checkCudaError(cudaMalloc(&d_input_sorted, imageSize * sizeof(uint8_t)), "cudaMalloc d_input_sorted");
    
    checkCudaError(cudaMemcpy(d_input, image_array, imageSize * sizeof(uint8_t), 
                             cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    checkCudaError(cudaMemcpy(d_input_sorted, image_array_sorted, imageSize * sizeof(uint8_t), 
                             cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    std::cout << "Image loaded and copied to device memory." << std::endl;

    // Open CSV file for writing results
    std::ofstream csvFile("memory_access_results.csv");
    if (!csvFile.is_open()) {
        std::cerr << "Error: Unable to create CSV file." << std::endl;
        return EXIT_FAILURE;
    }
    
    // Write CSV header
    csvFile << "ThreadsPerBlock;NumBlocks;CoalescedTime(ms);UncoalescedTime(ms)\n";

    // Performance testing
    std::cout << "\nRunning performance tests...\n" << std::endl;
    std::cout << std::setw(15) << "Threads/Block" 
              << std::setw(15) << "Blocks"
              << std::setw(15) << "Coalesced (ms)"
              << std::setw(15) << "Uncoalesced (ms)"
              << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (int threads = 32; threads <= 1024; threads += 1) {
        const int NUM_ITERATIONS = 10;
        TimingResult avgResult = {threads, 0, 0.0f, 0.0f};
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            TimingResult result = runKernelsWithTiming(d_input, d_output, d_input_sorted, pixelCount, threads);
            avgResult.coalescedTime += result.coalescedTime;
            avgResult.uncoalescedTime += result.uncoalescedTime;
            avgResult.numBlocks = result.numBlocks;
        }
        
        avgResult.coalescedTime /= NUM_ITERATIONS;
        avgResult.uncoalescedTime /= NUM_ITERATIONS;

        // Print results
        std::cout << std::setw(15) << avgResult.threadsPerBlock
                  << std::setw(15) << avgResult.numBlocks
                  << std::setw(15) << std::fixed << std::setprecision(3) << avgResult.coalescedTime
                  << std::setw(15) << std::fixed << std::setprecision(3) << avgResult.uncoalescedTime
                  << std::endl;

        // Write to CSV
        csvFile << avgResult.threadsPerBlock << ";"
                << avgResult.numBlocks << ";"
                << avgResult.coalescedTime << ";"
                << avgResult.uncoalescedTime << "\n";
    }

    csvFile.close();
    std::cout << "\nPerformance results saved to 'memory_access_results.csv'.\n";

    // Copy result back and save
    uint8_t* new_image_array = (uint8_t*)malloc(imageSize * sizeof(uint8_t));
    checkCudaError(cudaMemcpy(new_image_array, d_output, imageSize * sizeof(uint8_t), 
                             cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    
    save_image_array(new_image_array);

    free(image_array - OFFSET);
    free(new_image_array);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_sorted);
    
    return 0;
}