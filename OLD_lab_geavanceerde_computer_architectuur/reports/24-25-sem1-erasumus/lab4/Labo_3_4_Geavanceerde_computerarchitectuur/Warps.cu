/* 
 * Code snippet for importing / exporting image data.
 * 
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 * 
 */
#include <cstdint>      // Data types
#include <iostream>     // File operations

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

__global__ void invertVectorImage (uint8_t *inputVector,int numElements) {
    int stride = gridDim.x * blockDim.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = id;i<numElements;i+=stride){
        inputVector[i] = 255-inputVector[i];
    }

}

__global__ void invertVectorImageR (uint8_t *inputVector,int numElements) {
    int stride = gridDim.x * blockDim.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id%3==0){
        for(int i = id;i<numElements;i+=stride){
            inputVector[i] = 255-inputVector[i];
        }
    }
}


void invertImage(uint8_t* image_array_in,uint8_t* image_array,int elements,int threads,int blocks){
    
    
    uint8_t *d_array;
    
    cudaMalloc((void **)&d_array, elements * sizeof(uint8_t));
    

    cudaMemcpy(d_array,image_array_in, elements * sizeof(uint8_t), cudaMemcpyHostToDevice);

    double average = 0;

    for(int i = 0;i<3;i++){
        clock_t begin = clock();
        invertVectorImage<<<blocks, threads>>>(d_array,elements);
        clock_t end = clock();
        double time = (double) (end-begin) / CLOCKS_PER_SEC;
        average+=time;
    
    }
    printf("time;%f;",average/3);

    cudaMemcpy(image_array, d_array, elements*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    


    cudaFree (d_array);
    
}

void invertImageR(uint8_t* image_array_in,uint8_t* image_array,int elements,int threads,int blocks){
    uint8_t *d_array;
    
    cudaMalloc((void **)&d_array, elements * sizeof(uint8_t));
    
    double average = 0;
    cudaMemcpy(d_array,image_array_in, elements * sizeof(uint8_t), cudaMemcpyHostToDevice);
    for(int i =0;i<3;i++){
        clock_t begin = clock();
        invertVectorImageR<<<blocks, threads>>>(d_array,elements);
        clock_t end = clock();
        double time = (double) (end-begin) / CLOCKS_PER_SEC;
        average += time;
    }
    printf("timeR ;%f \n",average/3);

    cudaMemcpy(image_array, d_array, elements*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    


    cudaFree (d_array);
    
}

uint8_t* formatData(uint8_t* image_array,int elements){
    uint8_t* formatted_image_array = (uint8_t*)malloc(elements*sizeof(uint8_t));

    for(int i = 0;i<elements;i+=3){
            formatted_image_array[i/3] = image_array[i];
        if(i+1<elements){
            formatted_image_array[(i/3)+(elements/3)] = image_array[i+1];
        }
        if(i+2<elements){
            formatted_image_array[(i/3)+2*(elements/3)] = image_array[i+2];
        } 

    }
	
    return formatted_image_array;
    
}


int main (void) {
    
    // Read the image
    uint8_t* image_array = get_image_array();
    
    

    // Allocate output
    uint8_t* new_image_array = (uint8_t*)malloc(M*N*C);
    //start eigen code
    int elements = M*N*C; // number of elements in array
    uint8_t* formatted_image_array = formatData(image_array,elements);
   
    int blocks = 1;

    printf("length of input array = %d\n",M*N*C);

    

    for(int threads = 320;threads<512;threads++){
        int numthreads = threads * blocks;
        printf("threads; %d;",numthreads);
        printf("warps ;%d;",(threads +32-1)/32)*blocks;
        invertImage(formatted_image_array,new_image_array,elements,threads,blocks);
        invertImage(image_array, new_image_array,elements,threads,blocks); // call inversion function
        invertImageR(image_array,new_image_array,elements,threads,blocks);
    }

   
    // Save the image
    save_image_array(new_image_array);
    free(formatted_image_array);
    return 0;
  
}