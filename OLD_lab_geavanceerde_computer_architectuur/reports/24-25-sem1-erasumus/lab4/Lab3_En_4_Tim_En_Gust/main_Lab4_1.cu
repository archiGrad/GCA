#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <string>

#define M 960        // VR width
#define N 1280       // VR height
#define C 3          // Colors
#define OFFSET 16    // Header length
#define BLOCK_SIZE 192


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


void save_image_array(uint8_t* image_array, char* name){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen(name,"wb");
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

// Functions for reading/writing images (provided in base code)
uint8_t* get_image_array(void);
void save_image_array(uint8_t* image_array);

// Kernel voor uncoalesced memory access (RGBRGBRGB format)
__global__ void invertRedUncoalesced(uint8_t* image, uint8_t* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int deel = M * N;
    for (int i = idx; i < deel; i +=stride){
        if (idx < deel) {
            int base_idx = idx * 3;
            output[base_idx] = 255 - image[base_idx];       
            output[base_idx + 1] = image[base_idx + 1];     
            output[base_idx + 2] = image[base_idx + 2];     
        }
    }
}

// Kernel voor coalesced memory access (RRRR...GGGG...BBBB format)
__global__ void invertRedCoalesced(uint8_t* image, uint8_t* output) {
    int deel = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < deel; i +=stride){
        if (idx < deel) {
            output[idx] = 255 - image[idx];
            output[idx + deel] = image[idx + deel];
            output[idx + 2 * deel] = image[idx + 2 * deel];
        }
    }
}

//Zet om naar RRRR...GGGG...BBBB
void reformat(uint8_t* original, uint8_t* formatted, int size){
    int deel = M *N;
    for(int i = 0; i < size; i++){
        if(i%3 == 0){
            formatted[i/3] = original[i];
        }
        else if(i%3==1){
            formatted[(i/3)+deel] = original[i];
        }
        else{
            formatted[(i/3)+2*deel] = original[i];
        }
    }
}

//Zet om naar RGB...RGB...RGB
void unreformat(uint8_t* formatted, uint8_t* unformatted, int size){
    int deel = M * N;
    for(int i = 0; i < deel; i++){
        unformatted[i * 3] = formatted[i];                    // R
        unformatted[i * 3 + 1] = formatted[i + deel]; // G
        unformatted[i * 3 + 2] = formatted[i + 2 * deel]; // B
    }
}

void writeToCSV(int blocks, int threads, float time_ms, bool coalesced) {
    if(coalesced){
        std::ofstream file("output_c.csv", std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            return;
        }
        file << blocks * threads << "," << time_ms << "\n";
        file.close();
    }
    else{
        std::ofstream file("output_uc.csv", std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            return;
        }
        file << blocks * threads << "," << time_ms << "\n";
        file.close();
    }
}

int main() {
    uint8_t* h_image = get_image_array();
    int size = M * N * C;

    // Create output files and write headers
    std::ofstream file_c("output_c.csv");
    std::ofstream file_uc("output_uc.csv");
    if (!file_c.is_open() || !file_uc.is_open()) {
        std::cerr << "Failed to open output files" << std::endl;
        return 1;
    }
    file_c << "threads,time_ms\n";
    file_uc << "threads,time_ms\n";
    file_c.close();
    file_uc.close();

    for(int blocks = 1; blocks <= 32; blocks*=2) {
        float total_time_uc = 0;
        float total_time_c = 0;
        
        for(int i = 0; i < 100; i++) {
            uint8_t* h_output_coalesced = (uint8_t*)malloc(size);
            uint8_t* h_output_uncoalesced = (uint8_t*)malloc(size);
            uint8_t* h_temp_coalesced = (uint8_t*)malloc(size);
            
            uint8_t *d_image_coalesced, *d_image_uncoalesced, *d_output_coalesced, *d_output_uncoalesced;
            cudaMalloc(&d_image_coalesced, size);
            cudaMalloc(&d_image_uncoalesced, size);
            cudaMalloc(&d_output_coalesced, size);
            cudaMalloc(&d_output_uncoalesced, size);

            cudaMemcpy(d_image_uncoalesced, h_image, size, cudaMemcpyHostToDevice);
            
            reformat(h_image, h_temp_coalesced, size);
            cudaMemcpy(d_image_coalesced, h_temp_coalesced, size, cudaMemcpyHostToDevice);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            float ms;
            
            // Time uncoalesced kernel
            cudaEventRecord(start);
            invertRedUncoalesced<<<blocks, BLOCK_SIZE>>>(d_image_uncoalesced, d_output_uncoalesced);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            total_time_uc += ms;
            
            // Time coalesced kernel
            cudaEventRecord(start);
            invertRedCoalesced<<<blocks, BLOCK_SIZE>>>(d_image_coalesced, d_output_coalesced);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            total_time_c += ms; 
            
            cudaMemcpy(h_temp_coalesced, d_output_coalesced, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_output_uncoalesced, d_output_uncoalesced, size, cudaMemcpyDeviceToHost);
            
            unreformat(h_temp_coalesced, h_output_coalesced, size);
            
            if (i == 0) { 
                save_image_array(h_output_coalesced, (char*)"./coalesced_output.ppm");
                save_image_array(h_output_uncoalesced, (char*)"./uncoalesced_output.ppm");
            }
            
        
            cudaFree(d_image_coalesced);
            cudaFree(d_image_uncoalesced);
            cudaFree(d_output_coalesced);
            cudaFree(d_output_uncoalesced);
            free(h_output_coalesced);
            free(h_output_uncoalesced);
            free(h_temp_coalesced);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        writeToCSV(blocks, BLOCK_SIZE, total_time_uc/100, false);
        writeToCSV(blocks, BLOCK_SIZE, total_time_c/100, true);
    }
    
    free(h_image);
    return 0;
}