/* 
 * Code snippet for importing / exporting image data.
 * 
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 * 
 */
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h>

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 960       // VR width
#define N 1280       // VR height
#define C 3         // Colors
#define OFFSET 16   // Header length


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
    imageFile=fopen("./output_image.ppm","wb");
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

// Error checking helper function
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel to reverse RGB values
__global__ void reverseRGBKernel(uint8_t* input, uint8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = 255 - input[idx];
    }
}


int main (void) {

    
    // Read the image
    uint8_t* image_array = get_image_array();

    const int imageSize = M * N * C;
    std::cout << "Image size: " << imageSize << " bytes" << std::endl;

    // // Allocate output
    // uint8_t* new_image_array = (uint8_t*)malloc(M*N*C);
    // // Convert to grayscale using only the red color component
    // for(int i=0; i<M*N*C; i++){
    //     new_image_array[i] = image_array[i/3*3];
    // }
    
    // Allocate device memory
    uint8_t* d_input, *d_output;
    
    checkCudaError(cudaMalloc(&d_input, imageSize * sizeof(uint8_t)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, imageSize * sizeof(uint8_t)), "cudaMalloc d_output");
    
    // Copy image data to device
    std::cout << "Copying data to device..." << std::endl;
    checkCudaError(cudaMemcpy(d_input, image_array, imageSize * sizeof(uint8_t), 
                             cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocks = (M * N * C + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Grid size: " << blocks << " blocks of " << threadsPerBlock << " threads" << std::endl;
    
    reverseRGBKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, imageSize);

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "kernel launch");
    std::cout << "Kernel launched successfully" << std::endl;

    // Wait for kernel to finish
    checkCudaError(cudaDeviceSynchronize(), "kernel synchronize");
    std::cout << "Kernel execution completed" << std::endl;

    // Allocate host memory for output
    uint8_t* new_image_array = (uint8_t*)malloc(imageSize * sizeof(uint8_t));
    if (new_image_array == NULL) {
        std::cerr << "Failed to allocate host memory for output" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy output back to host
    checkCudaError(cudaMemcpy(new_image_array, d_output, imageSize * sizeof(uint8_t), 
                             cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Save the reversed image
    save_image_array(new_image_array);

    // Clean up
    free(image_array - OFFSET);
    free(new_image_array);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}