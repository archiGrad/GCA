#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda_runtime.h>
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include "chrono"
#include <fstream>

using namespace std::chrono;

#define N 683    // height
#define W1 512
#define W2 512
#define C 3         // Colors
#define OFFSET 15   // Header length


__global__ void get_diff(int *arr1, int *arr2, int* diffs, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 5 && idx % 3 == 0) {
        int diff_R = abs(arr1[idx] - arr2[0]) +abs(arr1[idx+3]-arr2[3]);
        int diff_G = abs(arr1[idx + 1] - arr2[1]) +abs(arr1[idx+4]-arr2[4]);
        int diff_B = abs(arr1[idx + 2] - arr2[2]) +abs(arr1[idx+5]-arr2[5]);
        int total_diff = diff_R + diff_G + diff_B;

        // Store the difference in the array
        diffs[idx / 3] = total_diff;
    }
    __syncthreads();
}

__global__ void get_minimum(int *diffs, int *min_value, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<size-1){
        atomicMin(min_value, diffs[idx]);
    }
    __syncthreads();
}

__global__ void get_index(int *diffs, int *min_value, int *min_index){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx<W1-1){
        if(diffs[idx]==*min_value){
            *min_index=idx;
        }
    }
}

void extract_rows(uint8_t* image_array, int** rows, int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                rows[y][(x * channels) + c] = static_cast<int>(image_array[((y * (width)) + x) * channels + c]);
            }
        }
    }
}

int mostFrequent(int* arr, int n) 
{ 
    int maxElement, count;
    int maxCount = 0;
    /* Frequency of each element is counted and checked.If it's greater than the utmost count element we found till now, then it is updated accordingly  */  
    for(int i = 0; i< n; i++)   //For loop to hold each element
    {
        count = 1;
        for(int j = i+1; j < n; j++)  //For loop to check for duplicate elements
        {
            if(arr[j] == arr[i])
            {
                count++;     //Increment  count
                /* If count of current element is more than 
                maxCount, then update maxElement */
                if(count > maxCount)
                {
                    maxElement = arr[j];
                    maxCount=count;
                }
            }
        }
    }
    return maxElement;
} 

uint8_t* get_image_array(const char* path, int W, int H, int colors){
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
    imageFile=fopen(path,"rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(W * H * colors * sizeof(uint8_t) + OFFSET);
    
    // Read the image
    fread(image_array, sizeof(uint8_t), W * H * colors * sizeof(uint8_t) + OFFSET, imageFile);
    
    // Close the file
    fclose(imageFile);
        
    // Calculate the size of the array
    size_t array_size = W * H * colors * sizeof(uint8_t) + OFFSET;
    std::cout << "Size of the array: " << array_size << " bytes" << std::endl;

    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}

void save_image_array(uint8_t* image_array, int width, int height, int colors){
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
    fprintf(imageFile,"%d %d\n", width, height);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
    
    // Write the image
    fwrite(image_array, 1, width*height*colors, imageFile);
    
    // Close the file
    fclose(imageFile);
}


int main() {
    srand(time(NULL)); // Seed random number generator
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (W1 * 3 + threadsPerBlock - 1) / threadsPerBlock;
    const int blocksPerGrid2 = (W1 + threadsPerBlock - 1) / threadsPerBlock;
    const int blocksPerGrid3 = (N + threadsPerBlock - 1) / threadsPerBlock;

    int min_array[N];
    //Initialize CPU variables
    int *min_diff;
    min_diff=(int *)malloc(sizeof(int)); *min_diff = 255; // Initialize with maximum possible difference
    int *min_index;
    min_index=(int *)malloc(sizeof(int)); *min_index = -1;
    int *tot_min_index;
    tot_min_index=(int *)malloc(sizeof(int)); *tot_min_index = 255;
    int *host_diffs = new int[W1];

    //Initialize GPU variables
    int *d_min_diff;
    int *d_min_index;
    int *d_tot_min_index;
    int *dev_diffs;
    int *dev_array1, *dev_array2;
    int *d_min_array;

    cudaMalloc((void**)&d_min_diff, sizeof(int));
    cudaMalloc((void**)&d_min_index, sizeof(int));
    cudaMalloc((void**)&d_tot_min_index, sizeof(int));
    cudaMemcpy(d_min_diff, min_diff, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_index, min_index, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tot_min_index, tot_min_index, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_array1, W1 * 3 * sizeof(int));
    cudaMalloc((void**)&dev_array2, W2 * 3 * sizeof(int));
    cudaMalloc((void**)&dev_diffs, W1 * sizeof(int));
    cudaMalloc((void**)&d_min_array, N*sizeof(int));

    
    uint8_t* image_array1 = get_image_array("./test1.ppm", W1, N, C);
    uint8_t* image_array2 = get_image_array("./test2.ppm", W2, N, C);

    int** rows = new int*[N];
    for (int i = 0; i < N; ++i) {
        rows[i] = new int[W1*3];
    }
    int** rows2 = new int*[N];
    for (int i = 0; i < N; ++i) {
        rows2[i] = new int[W2*3];
    }

    extract_rows(image_array1, rows, W1, N, C);
    extract_rows(image_array2, rows2, W2, N, C);

/*     std::cout << "Image array:" << std::endl;
    for (int i = 0; i < M*N*C; ++i) {
        std::cout << static_cast<int>(image_array[i]) << " ";
    }
    std::cout << std::endl; */

    std::cout << "Rows extracted" << std::endl;

     //Print rows
/*     std::cout << "Rows extracted from image:" << std::endl;
    for (int i = 1; i < 2; ++i) {
        for (int j = 0; j < 900; ++j) {
            std::cout << rows[i][j]<< " ";
        }
        std::cout << std::endl;
    }

    // Print rows2
    std::cout << "Rows2 extracted from image:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < 900; ++j) {
            std::cout << rows2[i][j] << " ";
        }
        std::cout << std::endl;
    } */
 

    for (int x=0; x<N; ++x){
    
        cudaMemcpy(dev_array1, rows[x], W1 * 3 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_array2, rows2[x], W2 * 3 * sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        get_diff<<<blocksPerGrid, threadsPerBlock>>>(dev_array1, dev_array2, dev_diffs, W1 * 3);

        cudaMemcpy(host_diffs, dev_diffs, W1 * sizeof(int), cudaMemcpyDeviceToHost);

        // Print out the full difference array
         std::cout << std::endl;
          std::cout << "Difference array:" << std::endl;
        for (int i = 0; i < W1; ++i) {
            std::cout << host_diffs[i] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl; 

        cudaDeviceSynchronize();

        get_minimum<<<blocksPerGrid2, threadsPerBlock>>>(dev_diffs, d_min_diff, W1);

        cudaDeviceSynchronize();

        get_index<<<blocksPerGrid2, threadsPerBlock>>> (dev_diffs, d_min_diff, d_min_index);

        cudaMemcpy(min_diff, d_min_diff, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(min_index, d_min_index, sizeof(int), cudaMemcpyDeviceToHost);

        if (*min_index != -1) {
            std::cout << "Minimum difference: " << *min_diff << " at index: " << *min_index << std::endl;
        } else {
            std::cout << "No valid minimum difference found!"<< *min_diff << *min_index << std::endl;
        } 
        min_array[x]=*min_index;
        //delete[] host_diffs;

    }
    cudaMemcpy(d_min_array, min_array, N*sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "Min array:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << min_array[i] << " " ;
    }
    std::cout << std::endl;

    *tot_min_index=mostFrequent(min_array, N);

    if (*tot_min_index != -1 && *tot_min_index<W1) {
            std::cout << "Total minimum difference at index to stitch: " << *tot_min_index << std::endl;
        } else {
            std::cout << "No valid total minimum difference found!" << std::endl;
            *tot_min_index=0;
    } 

    int width = W2+*tot_min_index;

    //Initialize stitched array
    uint8_t stitched_array[width*N*C];

     // Stitch the arrays row by row
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col <width; ++col){
            for (int c =0; c< C; ++c){
                if (col < *tot_min_index) {
                    // Copy elements from rows[0] up to min_index
                    stitched_array[(row * width * C) + (col * C) + c] = rows[row][col * C + c];
                } else {
                    // Copy elements from rows2
                    stitched_array[(row * width * C) + (col * C) + c] = rows2[row][(col-*tot_min_index)*C + c];
                }
            }
        }
    }

    std::cout << "Stitched array done!" << std::endl;

    // Save the stitched array
    save_image_array(stitched_array, width, N, C);
    std::cout << "Save image array done!" << std::endl;
   
    // Free memory
    cudaFree(dev_array1);
    cudaFree(dev_array2);
    cudaFree(dev_diffs);
     delete[] host_diffs;

    return 0;
}