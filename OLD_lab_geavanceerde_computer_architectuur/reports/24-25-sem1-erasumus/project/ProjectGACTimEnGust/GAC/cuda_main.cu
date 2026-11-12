#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric> 

const int INPUT_SIZE = 140;
const int LAYER1_OUTPUT = 140;
const int LAYER2_OUTPUT = 128;
const int LAYER3_OUTPUT = 64;
const int LAYER4_OUTPUT = 32;

__constant__ double d_bias_layer1[LAYER1_OUTPUT];
__constant__ double d_bias_layer2[LAYER2_OUTPUT];
__constant__ double d_bias_layer3[LAYER3_OUTPUT];
__constant__ double d_bias_layer4[LAYER4_OUTPUT];
__constant__ double d_bias_layer5[LAYER3_OUTPUT];
__constant__ double d_bias_layer6[LAYER2_OUTPUT];
__constant__ double d_bias_layer7[LAYER1_OUTPUT];
__constant__ double d_bias_layer8[LAYER1_OUTPUT];

// CUDA error checking macro
#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(e), __LINE__);  \
        exit(1);                                                    \
    }                                                               \
}

__global__ void matrixVectorMulKernel1(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer1[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

__global__ void matrixVectorMulKernel2(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer2[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

__global__ void matrixVectorMulKernel3(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer3[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

__global__ void matrixVectorMulKernel4(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer4[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

__global__ void matrixVectorMulKernel5(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer5[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

__global__ void matrixVectorMulKernel6(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer6[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

__global__ void matrixVectorMulKernel7(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer7[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

__global__ void matrixVectorMulKernel8(
    const double* input, 
    const double* weights, 
    double* output, 
    int inputSize, 
    int outputSize
) {
    int stride = blockDim.x * gridDim.x;
    
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < outputSize; j += stride) {
        double sum = d_bias_layer8[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[j * inputSize + i];
        }
        output[j] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void reluKernel(double* vector, int size) {
    int stride = blockDim.x * gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        vector[i] = fmax(0.0, vector[i]);
    }
}

// CUDA kernel for Sigmoid activation
__global__ void sigmoidKernel(double* vector, int size) {
    int stride = blockDim.x * gridDim.x; // Calculate the grid stride

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        vector[i] = 1.0 / (1.0 + exp(-vector[i]));
    }
}

// Utility function to load matrix from CSV
bool loadMatrix(const char* filename, std::vector<double>& matrix, int rows, int cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    matrix.clear();
    double value;
    std::string line;
    char delimiter = ';'; // Use semicolon as the default delimiter

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, delimiter)) {
            try {
                value = std::stod(cell);
                matrix.push_back(value);
            } catch (...) {
                std::cerr << "Error parsing value: " << cell << std::endl;
                return false;
            }
        }
    }

    if (matrix.size() != rows * cols) {
        std::cerr << "Warning: Expected " << rows * cols 
                  << " elements, but found " << matrix.size() 
                  << " in " << filename << std::endl;
        return false;
    }

    return true;
}

// Utility function to load vector from CSV
bool loadVector(const char* filename, std::vector<double>& vector, int expectedSize = -1) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    vector.clear();
    double value;
    std::string line, cell;
    char delimiter = ';'; // Use semicolon as the default delimiter

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, cell, delimiter)) {
            try {
                value = std::stod(cell);
                vector.push_back(value);
                if (expectedSize != -1 && vector.size() >= expectedSize) break;
            } catch (...) {
                std::cerr << "Error parsing value: " << cell << std::endl;
                return false;
            }
        }
    }

    if (expectedSize != -1 && vector.size() != expectedSize) {
        std::cerr << "Warning: Expected " << expectedSize 
                  << " elements, but found " << vector.size() 
                  << " in " << filename << std::endl;
        return false;
    }

    return true;
}

// Utility function to calculate Mean Squared Error
double calculateMSE(const double* input1, const double* input2, int size) {
    double mse = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = input1[i] - input2[i];
        mse += diff * diff;
    }
    return mse / size;
}

// Utility function to print vector
void printVector(const double* vector, int size, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < size; i++) {
        std::cout << vector[i] << std::endl;
    }
    std::cout << std::endl;
}

// Prints a device vector after copying it to the host
void printDeviceVector(const double* d_vector, int size, const char* name) {
    std::vector<double> h_vector(size);
    cudaMemcpy(h_vector.data(), d_vector, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    std::cout << name << ":\n";
    for (int i = 0; i < size; i++) {
        std::cout << h_vector[i] << "\n";
    }
    std::cout << "\n" << std::endl;
}

// Count anomalies based on a threshold
int countAnomalies(const double* input, const double* reconstructed, int size, double threshold) {
    int anomalyCount = 0;
    for (int i = 0; i < size; i++) {
        double diff = input[i] - reconstructed[i];
        if (diff * diff > threshold) {
            anomalyCount++;
        }
    }
    return anomalyCount;
}

int main() {
    // Layer configuration

    // Host vectors for input and outputs
    std::vector<double> h_input(INPUT_SIZE);
    std::vector<double> h_layer1_output(LAYER1_OUTPUT);
    std::vector<double> h_layer2_output(LAYER2_OUTPUT);
    std::vector<double> h_layer3_output(LAYER3_OUTPUT);
    std::vector<double> h_layer4_output(LAYER4_OUTPUT);
    std::vector<double> h_layer5_output(LAYER4_OUTPUT);
    std::vector<double> h_layer6_output(LAYER3_OUTPUT);
    std::vector<double> h_layer7_output(LAYER2_OUTPUT);
    std::vector<double> h_layer8_output(LAYER1_OUTPUT);
    std::vector<double> h_final_output(INPUT_SIZE);

    // Load input
    if (!loadVector("input_signal.csv", h_input, INPUT_SIZE)) {
        std::cerr << "Failed to load input signal" << std::endl;
        return 1;
    }

    // Load weights and biases for each layer
    std::vector<double> h_weights_layer1, h_bias_layer1;
    std::vector<double> h_weights_layer2, h_bias_layer2;
    std::vector<double> h_weights_layer3, h_bias_layer3;
    std::vector<double> h_weights_layer4, h_bias_layer4;
    std::vector<double> h_weights_layer5, h_bias_layer5;
    std::vector<double> h_weights_layer6, h_bias_layer6;
    std::vector<double> h_weights_layer7, h_bias_layer7;
    std::vector<double> h_weights_layer8, h_bias_layer8;

    // Load encoder weights and biases
    if (!loadMatrix("encoder_layer_1_weights.csv", h_weights_layer1, LAYER1_OUTPUT, INPUT_SIZE)) {
        std::cerr << "Failed to load layer 1 weights" << std::endl;
        return 1;
    }
    if (!loadVector("encoder_layer_1_biases.csv", h_bias_layer1, LAYER1_OUTPUT)) {
        std::cerr << "Failed to load layer 1 biases" << std::endl;
        return 1;
    }
    if (!loadMatrix("encoder_layer_3_weights.csv", h_weights_layer2, LAYER2_OUTPUT, LAYER1_OUTPUT)) {
        std::cerr << "Failed to load layer 2 weights" << std::endl;
        return 1;
    }
    if (!loadVector("encoder_layer_3_biases.csv", h_bias_layer2, LAYER2_OUTPUT)) {
        std::cerr << "Failed to load layer 2 biases" << std::endl;
        return 1;
    }
    if (!loadMatrix("encoder_layer_5_weights.csv", h_weights_layer3, LAYER3_OUTPUT, LAYER2_OUTPUT)) {
        std::cerr << "Failed to load layer 3 weights" << std::endl;
        return 1;
    }
    if (!loadVector("encoder_layer_5_biases.csv", h_bias_layer3, LAYER3_OUTPUT)) {
        std::cerr << "Failed to load layer 3 biases" << std::endl;
        return 1;
    }
    if (!loadMatrix("encoder_layer_7_weights.csv", h_weights_layer4, LAYER4_OUTPUT, LAYER3_OUTPUT)) {
        std::cerr << "Failed to load layer 4 weights" << std::endl;
        return 1;
    }
    if (!loadVector("encoder_layer_7_biases.csv", h_bias_layer4, LAYER4_OUTPUT)) {
        std::cerr << "Failed to load layer 4 biases" << std::endl;
        return 1;
    }

    // Load decoder weights and biases
    if (!loadMatrix("decoder_layer_1_weights.csv", h_weights_layer5, LAYER3_OUTPUT, LAYER4_OUTPUT)) {
        std::cerr << "Failed to load layer 5 weights" << std::endl;
        return 1;
    }
    if (!loadVector("decoder_layer_1_biases.csv", h_bias_layer5, LAYER3_OUTPUT)) {
        std::cerr << "Failed to load layer 5 biases" << std::endl;
        return 1;
    }
    if (!loadMatrix("decoder_layer_3_weights.csv", h_weights_layer6, LAYER2_OUTPUT, LAYER3_OUTPUT)) {
        std::cerr << "Failed to load layer 6 weights" << std::endl;
        return 1;
    }
    if (!loadVector("decoder_layer_3_biases.csv", h_bias_layer6, LAYER2_OUTPUT)) {
        std::cerr << "Failed to load layer 6 biases" << std::endl;
        return 1;
    }
    if (!loadMatrix("decoder_layer_5_weights.csv", h_weights_layer7, LAYER1_OUTPUT, LAYER2_OUTPUT)) {
        std::cerr << "Failed to load layer 7 weights" << std::endl;
        return 1;
    }
    if (!loadVector("decoder_layer_5_biases.csv", h_bias_layer7, LAYER1_OUTPUT)) {
        std::cerr << "Failed to load layer 7 biases" << std::endl;
        return 1;
    }
    if (!loadMatrix("decoder_layer_7_weights.csv", h_weights_layer8, LAYER1_OUTPUT, LAYER1_OUTPUT)) {
        std::cerr << "Failed to load layer 8 weights" << std::endl;
        return 1;
    }
    if (!loadVector("decoder_layer_7_biases.csv", h_bias_layer8, LAYER1_OUTPUT)) {
        std::cerr << "Failed to load layer 8 biases" << std::endl;
        return 1;
    }

    // Allocate device memory
    double *d_input, *d_layer1_output, *d_layer2_output, *d_layer3_output, *d_layer4_output;
    double *d_layer5_output, *d_layer6_output, *d_layer7_output, *d_layer8_output, *d_final_output;
    double *d_weights_layer1;
    double *d_weights_layer2;
    double *d_weights_layer3;
    double *d_weights_layer4;
    double *d_weights_layer5;
    double *d_weights_layer6;
    double *d_weights_layer7;
    double *d_weights_layer8;

    // Allocate device memory
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_layer1_output, LAYER1_OUTPUT * sizeof(double));
    cudaMalloc(&d_layer2_output, LAYER2_OUTPUT * sizeof(double));
    cudaMalloc(&d_layer3_output, LAYER3_OUTPUT * sizeof(double));
    cudaMalloc(&d_layer4_output, LAYER4_OUTPUT * sizeof(double));
    cudaMalloc(&d_layer5_output, LAYER2_OUTPUT * sizeof(double));
    cudaMalloc(&d_layer6_output, LAYER1_OUTPUT * sizeof(double));
    cudaMalloc(&d_layer7_output, LAYER1_OUTPUT * sizeof(double));
    cudaMalloc(&d_layer8_output, LAYER1_OUTPUT * sizeof(double));
    cudaMalloc(&d_final_output, INPUT_SIZE * sizeof(double));

    cudaMalloc(&d_weights_layer1, LAYER1_OUTPUT * INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_weights_layer2, LAYER2_OUTPUT * LAYER1_OUTPUT * sizeof(double));
    cudaMalloc(&d_weights_layer3, LAYER3_OUTPUT * LAYER2_OUTPUT * sizeof(double));
    cudaMalloc(&d_weights_layer4, LAYER4_OUTPUT * LAYER3_OUTPUT * sizeof(double));
    cudaMalloc(&d_weights_layer5, LAYER3_OUTPUT * LAYER4_OUTPUT * sizeof(double));
    cudaMalloc(&d_weights_layer6, LAYER2_OUTPUT * LAYER3_OUTPUT * sizeof(double));
    cudaMalloc(&d_weights_layer7, LAYER1_OUTPUT * LAYER2_OUTPUT * sizeof(double));
    cudaMalloc(&d_weights_layer8, LAYER1_OUTPUT * LAYER1_OUTPUT * sizeof(double));

    // Copy input and weights to device
    cudaMemcpy(d_input, h_input.data(), INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    //Allocate Biases on constant memory
    cudaMemcpyToSymbol(d_bias_layer1, h_bias_layer1.data(), LAYER1_OUTPUT * sizeof(double));
    cudaMemcpyToSymbol(d_bias_layer2, h_bias_layer2.data(), LAYER2_OUTPUT * sizeof(double));
    cudaMemcpyToSymbol(d_bias_layer3, h_bias_layer3.data(), LAYER3_OUTPUT * sizeof(double));
    cudaMemcpyToSymbol(d_bias_layer4, h_bias_layer4.data(), LAYER4_OUTPUT * sizeof(double));

    cudaMemcpyToSymbol(d_bias_layer5, h_bias_layer5.data(), LAYER3_OUTPUT * sizeof(double));
    cudaMemcpyToSymbol(d_bias_layer6, h_bias_layer6.data(), LAYER2_OUTPUT * sizeof(double));
    cudaMemcpyToSymbol(d_bias_layer7, h_bias_layer7.data(), LAYER1_OUTPUT * sizeof(double));
    cudaMemcpyToSymbol(d_bias_layer8, h_bias_layer8.data(), LAYER1_OUTPUT * sizeof(double));

    
    // Encoder layer weights
    cudaMemcpy(d_weights_layer1, h_weights_layer1.data(), LAYER1_OUTPUT * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_layer2, h_weights_layer2.data(), LAYER2_OUTPUT * LAYER1_OUTPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_layer3, h_weights_layer3.data(), LAYER3_OUTPUT * LAYER2_OUTPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_layer4,h_weights_layer4.data(), LAYER4_OUTPUT * LAYER3_OUTPUT * sizeof(double), cudaMemcpyHostToDevice);

    // Decoder layer weights
    cudaMemcpy(d_weights_layer5, h_weights_layer5.data(), LAYER3_OUTPUT * LAYER4_OUTPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_layer6, h_weights_layer6.data(), LAYER2_OUTPUT * LAYER3_OUTPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_layer7, h_weights_layer7.data(), LAYER1_OUTPUT * LAYER2_OUTPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_layer8, h_weights_layer8.data(), LAYER1_OUTPUT * LAYER1_OUTPUT * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    const int THREADS_PER_BLOCK = 64;
    int blocks_layer1 =3, blocks_layer2=3, blocks_layer3=3, blocks_layer4=3, blocks_layer5=3, blocks_layer6=3, blocks_layer7=3, blocks_layer8 = 3;
    const int ITERATIONS = 10;
    std::vector<float> duration_ms(ITERATIONS-1); // Store individual durations

    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        cudaEvent_t start_cuda, stop_cuda;
        cudaEventCreate(&start_cuda);
        cudaEventCreate(&stop_cuda);

        cudaEventRecord(start_cuda);

        // Forward pass: Encoder
        matrixVectorMulKernel1<<<blocks_layer1, THREADS_PER_BLOCK>>>(d_input, d_weights_layer1, d_layer1_output, INPUT_SIZE, LAYER1_OUTPUT);
        cudaCheckError();
        reluKernel<<<blocks_layer1, THREADS_PER_BLOCK>>>(d_layer1_output, LAYER1_OUTPUT);
        cudaCheckError();

        matrixVectorMulKernel2<<<blocks_layer2, THREADS_PER_BLOCK>>>(d_layer1_output, d_weights_layer2, d_layer2_output, LAYER1_OUTPUT, LAYER2_OUTPUT);
        cudaCheckError();
        reluKernel<<<blocks_layer2, THREADS_PER_BLOCK>>>(d_layer2_output, LAYER2_OUTPUT);
        cudaCheckError();

        matrixVectorMulKernel3<<<blocks_layer3, THREADS_PER_BLOCK>>>(d_layer2_output, d_weights_layer3, d_layer3_output, LAYER2_OUTPUT, LAYER3_OUTPUT);
        cudaCheckError();
        reluKernel<<<blocks_layer3, THREADS_PER_BLOCK>>>(d_layer3_output, LAYER3_OUTPUT);
        cudaCheckError();

        matrixVectorMulKernel4<<<blocks_layer4, THREADS_PER_BLOCK>>>(d_layer3_output, d_weights_layer4, d_layer4_output, LAYER3_OUTPUT, LAYER4_OUTPUT);
        cudaCheckError();

        // Forward pass: Decoder
        matrixVectorMulKernel5<<<blocks_layer5, THREADS_PER_BLOCK>>>(d_layer4_output, d_weights_layer5, d_layer5_output, LAYER4_OUTPUT, LAYER3_OUTPUT);
        cudaCheckError();
        reluKernel<<<blocks_layer5, THREADS_PER_BLOCK>>>(d_layer5_output, LAYER2_OUTPUT);
        cudaCheckError();

        matrixVectorMulKernel6<<<blocks_layer6, THREADS_PER_BLOCK>>>(d_layer5_output, d_weights_layer6, d_layer6_output, LAYER3_OUTPUT, LAYER2_OUTPUT);
        cudaCheckError();
        reluKernel<<<blocks_layer6, THREADS_PER_BLOCK>>>(d_layer6_output, LAYER1_OUTPUT);
        cudaCheckError();

        matrixVectorMulKernel7<<<blocks_layer7, THREADS_PER_BLOCK>>>(d_layer6_output, d_weights_layer7, d_layer7_output, LAYER2_OUTPUT, LAYER1_OUTPUT);
        cudaCheckError();
        reluKernel<<<blocks_layer7, THREADS_PER_BLOCK>>>(d_layer7_output, LAYER1_OUTPUT);
        cudaCheckError();

        matrixVectorMulKernel8<<<blocks_layer8, THREADS_PER_BLOCK>>>(d_layer7_output, d_weights_layer8, d_layer8_output, LAYER1_OUTPUT, LAYER1_OUTPUT);
        cudaCheckError();
        sigmoidKernel<<<blocks_layer8, THREADS_PER_BLOCK>>>(d_layer8_output, LAYER1_OUTPUT);
        cudaCheckError();

        cudaDeviceSynchronize();
        cudaEventRecord(stop_cuda);
        cudaEventSynchronize(stop_cuda);

        float ms = 0;
        cudaEventElapsedTime(&ms, start_cuda, stop_cuda);
        if (iteration != 0) {
            duration_ms[iteration - 1] = ms; 
        }

        cudaEventDestroy(start_cuda);
        cudaEventDestroy(stop_cuda);
    }

    // Calculate average and display timing results
    float total_ms = std::accumulate(duration_ms.begin(), duration_ms.end(), 0.0f);
    float average_ms = total_ms / ITERATIONS;

    std::cout << "Timing for " << ITERATIONS-1 << " iterations: ";
    for (float t : duration_ms) {
        std::cout << t << " ms ";
    }
    std::cout << "\nAverage execution time: " << average_ms << " ms\n";

    // Copy final output back to host
    cudaMemcpy(h_final_output.data(), d_layer8_output, INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Calculate Mean Squared Error between input and reconstructed signal
    double mse = calculateMSE(h_input.data(), h_final_output.data(), INPUT_SIZE);
    std::cout << "Mean Squared Error: " << mse << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_layer1_output);
    cudaFree(d_layer2_output);
    cudaFree(d_layer3_output);
    cudaFree(d_layer4_output);
    cudaFree(d_layer5_output);
    cudaFree(d_layer6_output);
    cudaFree(d_layer7_output);
    cudaFree(d_layer8_output);
    cudaFree(d_final_output);
    cudaFree(d_weights_layer1);
    cudaFree(d_weights_layer2);
    cudaFree(d_weights_layer3);
    cudaFree(d_weights_layer5);
    cudaFree(d_weights_layer6);
    cudaFree(d_weights_layer7);
    cudaFree(d_weights_layer8);

    double threshold = mse * 2; //adjustable
    int anomalyCount = countAnomalies(h_input.data(), h_final_output.data(), INPUT_SIZE, threshold);
    std::cout << "Number of anomalies: " << anomalyCount << std::endl;

    return 0;
}