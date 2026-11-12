#include <vector>
#include <device_launch_parameters.h>
#include <cstdint>      
#include <iostream>     
#include <cuda_runtime.h>
#include <iomanip>      

#define M 960       
#define N 1280      
#define C 3         
#define OFFSET 16 

// dus 2 kernels, 1 voor de R component van de pixels en 1 voor de G EN B componenten van de pixels
// Is het dan zo dat de kernel voor de GB component ook 2 keer wordt uitgevoerd? 1 keer voor de G en 1 keer voor de B? of dubbel zo hard belast wordt?
// De R kernel en de GB kernel worden sequentieel uitgevoerd
// Enkel binnen een kernel worden de threads parallel uitgevoerd
// verschil met de vorige code is dat je hier de kernel 2 keer uitvoert, 1 keer voor de R component en 1 keer voor de GB componenten
// sneller omdat de threads binnen een warp niet moeten wachten op elkaar, want elke thread binnen een warp doet dezelfde operatie
// modulo sneller dan 255 - waarde?

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
    return image_array + OFFSET;  // waarom doe je dit? waarom niet gewoon image_array?
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


// Separate kernel for red channel processing
__global__ void processRedChannelKernel(uint8_t* input, uint8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Process only red channel values
        uint8_t redValue = input[idx * 3]; // Multiply by 3 to get to red positions
        if (redValue >= 100 && redValue <= 200) {
            output[idx * 3] = (redValue % 25) * 10;
        } else {
            output[idx * 3] = 255 - redValue;
        }
    }
}

// Separate kernel for green and blue channels
__global__ void processGBChannelsKernel(uint8_t* input, uint8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Process green and blue channels
        int baseIdx = idx * 2; // Two components per pixel (G and B)
        
        // Adjust input/output indices to account for RGB structure
        int inputIdx = baseIdx + idx + 1;  // +1 to skip red positions
        int outputIdx = inputIdx;
        
        // Process both G and B values
        output[outputIdx] = 255 - input[inputIdx];
        if (baseIdx + 1 < size * 2) { // Ensure we don't go out of bounds
            output[outputIdx + 1] = 255 - input[inputIdx + 1];
        }
    }
}

// Modified timing structure to accommodate multiple kernels
struct TimingResult {
    int threadsPerBlock;
    int numBlocks;
    float redKernelTime;
    float gbKernelTime;
    float totalTime;
};

// Function to run kernels with timing
TimingResult runKernelsWithTiming(uint8_t* d_input, uint8_t* d_output, int imageSize, int threadsPerBlock) {
    TimingResult result;
    result.threadsPerBlock = threadsPerBlock;
    
    // klopt het dat de redPixelCount en gbPixelCount gelijk zijn? lijkt me onlogisch want de redPixelCount is een derde van de imageSize, dus de gbPixelCount zou twee derde van de imageSize moeten zijn?
    // Calculate sizes for each kernel
    int redPixelCount = imageSize / 3;  // Total pixels (each has RGB)
    int gbPixelCount = redPixelCount;   // Same number of pixels for GB
    //int gbPixelCount = redPixelCount * 2;   ????
    
    // Calculate blocks for each kernel
    result.numBlocks = (redPixelCount + threadsPerBlock - 1) / threadsPerBlock;
    
    // Create CUDA events for timing
    cudaEvent_t startRed, stopRed, startGB, stopGB;
    cudaEventCreate(&startRed);
    cudaEventCreate(&stopRed);
    cudaEventCreate(&startGB);
    cudaEventCreate(&stopGB);
    
    // Time red channel kernel
    cudaEventRecord(startRed);
    processRedChannelKernel<<<result.numBlocks, threadsPerBlock>>>(d_input, d_output, redPixelCount);
    cudaEventRecord(stopRed);
    
    // Time GB channels kernel
    cudaEventRecord(startGB);
    processGBChannelsKernel<<<result.numBlocks, threadsPerBlock>>>(d_input, d_output, gbPixelCount);
    cudaEventRecord(stopGB);
    
    // Synchronize and calculate times
    cudaEventSynchronize(stopRed);
    cudaEventSynchronize(stopGB);
    cudaEventElapsedTime(&result.redKernelTime, startRed, stopRed);
    cudaEventElapsedTime(&result.gbKernelTime, startGB, stopGB);
    
    result.totalTime = result.redKernelTime + result.gbKernelTime;
    
    // Cleanup
    cudaEventDestroy(startRed);
    cudaEventDestroy(stopRed);
    cudaEventDestroy(startGB);
    cudaEventDestroy(stopGB);
    
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

    // Performance testing
    std::cout << "\nRunning performance tests...\n" << std::endl;
    std::cout << std::setw(15) << "Threads/Block" 
              << std::setw(15) << "Blocks"
              << std::setw(15) << "Red (ms)"
              << std::setw(15) << "GB (ms)"
              << std::setw(15) << "Total (ms)"
              << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (int threads = 32; threads <= 1024; threads += 2) {  // hier start je weer van 32, terwijl je in de vorige code van 2 startte?
        const int NUM_ITERATIONS = 10;
        TimingResult avgResult = {threads, 0, 0.0f, 0.0f, 0.0f};
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            TimingResult result = runKernelsWithTiming(d_input, d_output, imageSize, threads);
            avgResult.redKernelTime += result.redKernelTime;
            avgResult.gbKernelTime += result.gbKernelTime;
            avgResult.numBlocks = result.numBlocks;
        }
        
        // Calculate averages
        avgResult.redKernelTime /= NUM_ITERATIONS;
        avgResult.gbKernelTime /= NUM_ITERATIONS;
        avgResult.totalTime = avgResult.redKernelTime + avgResult.gbKernelTime;
        
        // Print results
        std::cout << std::setw(15) << avgResult.threadsPerBlock
                  << std::setw(15) << avgResult.numBlocks
                  << std::setw(15) << std::fixed << std::setprecision(3) << avgResult.redKernelTime
                  << std::setw(15) << avgResult.gbKernelTime
                  << std::setw(15) << avgResult.totalTime
                  << std::endl;
        
        timingResults.push_back(avgResult);
    }

    // Find the best configuration
    auto bestConfig = std::min_element(timingResults.begin(), timingResults.end(),
        [](const TimingResult& a, const TimingResult& b) {
            return a.totalTime < b.totalTime;
        });

    std::cout << "\nBest configuration:" << std::endl;
    std::cout << "Threads per block: " << bestConfig->threadsPerBlock << std::endl;
    std::cout << "Number of blocks: " << bestConfig->numBlocks << std::endl;
    std::cout << "Red kernel time: " << bestConfig->redKernelTime << " ms" << std::endl;
    std::cout << "GB kernel time: " << bestConfig->gbKernelTime << " ms" << std::endl;
    std::cout << "Total time: " << bestConfig->totalTime << " ms" << std::endl;

    // Copy result back and save
    uint8_t* new_image_array = (uint8_t*)malloc(imageSize * sizeof(uint8_t));
    checkCudaError(cudaMemcpy(new_image_array, d_output, imageSize * sizeof(uint8_t), 
                             cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    
    // klopt het dat dit het resultaat is van de laatste iteratie? (voor het gemiddelde te bepalen)
    save_image_array(new_image_array);

    // Cleanup
    free(image_array - OFFSET);
    free(new_image_array);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}