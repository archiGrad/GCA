/* 
 * Code snippet for importing / exporting image data.
 * 
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 * 
 */
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <stdint.h>
#include <fstream>
#include <string>

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
    imageFile=fopen("./gg.ppm","wb");
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

__global__ void invertPhoto(uint8_t* array, int blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max = M*N*C;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < max; i +=stride){
        if(i % 3 == 0 && array[i] >= 100 && array[i] <= 200){
            array[i] = (array[i]%25) * 10;
        } 
        else {
            array[i] = 255 - array[i];
        }
    }

    /*for (int i = idx; i < max; i +=stride){
        array[i] = 255;
    }*/
}

__global__ void invertPhotoWithoutDivergence(uint8_t* array, int blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max = M*N*C;
    int stride = blockDim.x * gridDim.x;
    int deel = max/3;

    for (int i = idx; i < max; i +=stride){
        if(i < deel && array[i] >= 100 && array[i] <= 200){
            array[i] = (array[i]%25) * 10;
        } 
        else if(i >= deel){
            array[i] = 255 - array[i];
        }
    }

    /*for (int i = idx; i < max; i +=stride){
        array[i] = 255;
    }*/
}



void writeToCSV(int blocks, int threads, int time_ms) {
    // Open the file in append mode
    std::ofstream file("output.csv", std::ios::app);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return;
    }

    // Write the data in CSV format
    file << blocks * threads << "," << time_ms << "\n";

    // Close the file
    file.close();
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

int main (void) {
    // Read the image
    uint8_t* image_array = get_image_array();

    // Allocate output
    uint8_t* new_image_array = (uint8_t*)malloc(M*N*C);
    int size = M*N*C;
    
    // Convert to grayscale using only the red color component
    /*for(int i=0; i<M*N*C; i++){
        new_image_array[i] = image_array[i/3*3];
    }*/
    
    bool divergence = true;
    if(divergence){
        uint8_t* formatted = (uint8_t*)malloc(M*N*C);
        formatted = reformat(image_array, formatted, size);
    }

    for(int threads = 32; threads <= 1024; threads+=32) {
        std::cout << threads;
        for(int blocks = 1; blocks <= 32; blocks*=2) {
            std::cout << "-------------------------------------------------------" << std::endl;
            std::cout << "blocks: " << blocks << " threads: " << threads << std::endl;
            float total_time = 0;
            for(int i = 0; i < 100; i++){
                uint8_t* outputArray;
                cudaMalloc(&outputArray, sizeof(uint8_t) * size);

                if(divergence) {cudaMemcpy(outputArray, formatted, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);}
                else {cudaMemcpy(outputArray, image_array, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);}
                
                cudaEvent_t start_cuda, stop_cuda;
                cudaEventCreate(&start_cuda);
                cudaEventCreate(&stop_cuda);
                cudaEventRecord(start_cuda);

                //1 -> number of blocks; 2 -> threads per block
                if(divergence) {invertPhotoWithoutDivergence<<<blocks, threads>>>(outputArray, blocks);}
                else{ invertPhoto<<<blocks, threads>>>(outputArray, blocks);}
                
                cudaMemcpy(new_image_array, outputArray, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost);
                
                cudaDeviceSynchronize();
                cudaEventRecord(stop_cuda);
                cudaEventSynchronize(stop_cuda);
                float ms;
                cudaEventElapsedTime(&ms, start_cuda, stop_cuda);
                total_time += ms;

                // Save the image
                save_image_array(new_image_array);
                //save_image_array(image_array);
                cudaFree(outputArray);
            }
            writeToCSV(blocks, threads, total_time/100)
            std::cout << total_time/100 << "ms elapsed";
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    free(new_image_array);
    
    return 0;
}