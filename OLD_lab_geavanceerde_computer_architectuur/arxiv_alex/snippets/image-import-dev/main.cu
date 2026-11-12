// https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/

#include <stdio.h>
#include <iostream>


#define WARMUP_CYCLES 10
#define AVERAGING_CYCLES 101

#define M 32
#define N 8
#define IMG_SIZE M*N*3


__global__ void invert(u_int8_t* img, u_int32_t size){
    u_int16_t tid = threadIdx.x;
    u_int32_t id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < size){
        img[id] = 255 - img[id];
    }
}


float time_the_kernel(void (*kernel)(u_int8_t*, u_int32_t), int blk, int tpb, u_int8_t* d_img, u_int32_t size ){
    
    // Run a warmup - needed on some machines (GPU not used for graphics driver, hence, needs warmup?)
    for(int i=0; i<WARMUP_CYCLES; i++){
        kernel<<<blk, tpb>>>(d_img, size);
    }

    float milliseconds_avg = 0;

    // Time the code
    for(int i=0; i<AVERAGING_CYCLES; i++){

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        kernel<<<blk, tpb>>>(d_img, size);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        milliseconds_avg += milliseconds / AVERAGING_CYCLES;
    }

    return milliseconds_avg;
}


int main (void) {

    u_int8_t h_img[M*N];
    u_int8_t h_img_res[M*N];
    u_int8_t* d_img;
    for (int i=0; i<M*N; i++){
        h_img[i] = i;
    }

    
    cudaMalloc(&d_img, M*N*sizeof(u_int8_t));
    cudaMemcpy( d_img, &h_img, M*N*sizeof(u_int8_t), cudaMemcpyHostToDevice);

    // invert<<<1,8>>>(d_img, M*N);

    float timing = time_the_kernel(&invert, 1, 256, d_img, M*N*sizeof(u_int8_t));

    std::cout << "timing: " << timing << std::endl;

    cudaDeviceSynchronize();

    cudaMemcpy( &h_img_res, d_img, M*N*sizeof(u_int8_t), cudaMemcpyDeviceToHost);

    
    


    // https://stackoverflow.com/questions/36288421/c-create-png-bitmap-from-array-of-numbers

    FILE *imageFile;
    imageFile=fopen("./image.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel

    unsigned char* pix = (unsigned char*)malloc(M*N*3*sizeof(unsigned char));
    for(int i=0; i<M*N*3; i++){
        pix[i] = (unsigned char)(i/3);
        std::cout << "Val: " << +pix[i] << std::endl;
    }
    fwrite(pix, 1, M*N*3, imageFile);
    fclose(imageFile);



    imageFile=fopen("./con.ppm","rb");

    int offset = 12;
    unsigned char* pix_read = (unsigned char*)malloc(M*N*3*sizeof(unsigned char)+offset);
    fread(pix_read, sizeof(unsigned char), M*N*3*sizeof(unsigned char)+offset, imageFile);
    for(int i=0; i<M*N*3; i++){
        std::cout << +pix_read[i+offset] << " ";
    }
     
    return 0;
}
