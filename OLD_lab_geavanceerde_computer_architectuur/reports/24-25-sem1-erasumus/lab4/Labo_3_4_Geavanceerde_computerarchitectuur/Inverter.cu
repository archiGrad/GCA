/* 
 * Code snippet for importing / exporting image data.
 * 
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 * 
 */
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <chrono>
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
     *  - The first 3*M data elements represent the firts row of the image
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


void save_image_array(uint8_t* image_array, const char* pad){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen(pad,"wb");
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


__global__ void InvertRed_RGB(uint8_t* image, uint8_t* output, int width, int height){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_count = width*height;
    int stride = 3;

    if(idx < pixel_count){
        int red_index = idx*stride;
        output[red_index] = 255 - image[red_index];
        output[red_index+1] = image[red_index+1];
        output[red_index+2] = image[red_index+2];
    }
}

__global__ void invertRed_RRGGBB(uint8_t* image, uint8_t* output, int width, int height){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_count = width*height;

    if(idx < pixel_count){
        int red_index = idx;
        int green_index = pixel_count + idx;
        int blue_index = 2*pixel_count + idx;

        output[red_index] = 255 - image[red_index];
        output[green_index] = image[green_index];
        output[blue_index] = image[blue_index];
    }
}

void invertRedAndTime(uint8_t* image, uint8_t* output, bool is_interleaved){
    int image_size = M*N*C;

    uint8_t *d_image, *d_output;
    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_output, image_size);

    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (M*N+ blockSize -1)/ blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    if(is_interleaved){
        InvertRed_RGB<<<gridSize, blockSize>>>(d_image, d_output, M, N);
        printf("RGB;");
    } else{
        invertRed_RRGGBB<<<gridSize, blockSize>>>(d_image, d_output, M, N);
        printf("RRGGBB;");
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    printf("Execution time; %f ms \n", duration);

    cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_output);

}

void interleaved_to_planar(uint8_t* interleaved_image, uint8_t* planar_image, int width, int height){ 
    int pixel_count = width * height; // Planar image structure: first R, then G, then B 
    for (int i = 0; i < pixel_count; ++i){ // Red components: stored at the first part of the array 
    planar_image[i] = interleaved_image[i * 3]; // R channel 
    planar_image[pixel_count + i] = interleaved_image[i * 3 + 1]; // G channel 
    planar_image[2 * pixel_count + i] = interleaved_image[i * 3 + 2]; // B channel 
    } 
}


void planar_to_interleaved(uint8_t* planar_image, uint8_t* interleaved_image, int width, int height) { 
    int pixel_count = width * height; 
    for (int i = 0; i < pixel_count; ++i) { 
        interleaved_image[i * 3] = planar_image[i]; // R channel 
        interleaved_image[i * 3 + 1] = planar_image[pixel_count + i]; // G channel 
        interleaved_image[i * 3 + 2] = planar_image[2 * pixel_count + i]; // B channel 
    } 
}


uint8_t* generate_random_image(int width, int height){
    int pixel_count = width*height*C;
    uint8_t* image_array = new uint8_t[pixel_count];

    srand(time(NULL));

    for(int i=0; i<pixel_count; i++){
        image_array[i] = rand()%255;
    }
    return image_array;
}

int main (void) {
    
    // Read the image
    uint8_t* image_array = generate_random_image(M,N);
    
    // Allocate output
    uint8_t* output_RGB = (uint8_t*)malloc(M*N*C);
    uint8_t* input_RRGGBB = (uint8_t*)malloc(M*N*C);
    uint8_t* output_RRGGBB = (uint8_t*)malloc(M*N*C);
    uint8_t* RRGGBB = (uint8_t*)malloc(M*N*C);

    int block_size[] = {1,2,4,8,16,32,64,128, 256, 512,1024};
    int num_sizes = sizeof(block_size)/sizeof(block_size[0]);

    for( int i =0; i < num_sizes; i++){
        int gridSize = block_size[i];
        printf("Testing with block size; %d\n", gridSize);
        //printf("Processing RGB format: \n");
        invertRedAndTime(image_array, output_RGB, true);
        //printf("Processing RR..RGG...GB...B format... \n");
        interleaved_to_planar(image_array, input_RRGGBB, M, N);
        invertRedAndTime(input_RRGGBB, output_RRGGBB, false);
        planar_to_interleaved(output_RRGGBB, RRGGBB,N,M);
        // Save the image
        save_image_array(output_RGB, "./RGB.ppm");
        save_image_array(output_RRGGBB, "./RRGGBB.ppm");
    }

    delete[] image_array;
    delete[] output_RGB;
    delete[] output_RRGGBB;    
    return 0;
}
