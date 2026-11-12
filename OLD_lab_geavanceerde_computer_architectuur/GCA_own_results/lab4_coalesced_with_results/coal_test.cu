#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// note: results of first run might be misleading due to caching!!


//If the CUDA call fails, it prints: File name, Line number, Error code, Human-readable error string
//it prevents silent failures.
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ \
                  << ", code: " << error \
                  << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ================= CPU IMPLEMENTATIONS =================

void cpu_copy(const float* A, float* B, int N)
{
    int total = N * N;
    for (int i = 0; i < total; i++)
        B[i] = A[i];
}

void cpu_transpose(const float* A, float* B, int N)
{
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
            B[col * N + row] = A[row * N + col];
}

// ================= GPU KERNELS =================

__global__ void gpu_copy(const float* A, float* B, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        B[idx] = A[idx];
}

__global__ void gpu_transpose(const float* A, float* B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (idx < total) {
        int row = idx / N;
        int col = idx % N;

        B[col * N + row] = A[row * N + col];
    }
}

// ================= MAIN =================

int main(int argc, char** argv)
{
    int N = 2048;
    if (argc > 1)
        N = atoi(argv[1]);

    size_t bytes = N * N * sizeof(float);
    int total = N * N;

    std::cout << "Matrix size: " << N << " x " << N << std::endl;

    // Host memory
    std::vector<float> h_A(total);
    std::vector<float> h_B(total);
    std::vector<float> h_B_ref(total);

    // Initialize
    for (int i = 0; i < total; i++)
        h_A[i] = static_cast<float>(i);

    // ================= CPU COPY =================
    auto start = std::chrono::high_resolution_clock::now();
    cpu_copy(h_A.data(), h_B_ref.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    double cpu_copy_time =
        std::chrono::duration<double>(end - start).count();

    std::cout << "CPU Copy Time: " << cpu_copy_time << " s\n";

    // ================= CPU TRANSPOSE =================
    start = std::chrono::high_resolution_clock::now();
    cpu_transpose(h_A.data(), h_B_ref.data(), N);
    end = std::chrono::high_resolution_clock::now();

    double cpu_transpose_time =
        std::chrono::duration<double>(end - start).count();

    std::cout << "CPU Transpose Time: " << cpu_transpose_time << " s\n";

    // ================= DEVICE MEMORY =================
    float *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));

    CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // ================= GPU COPY =================
    cudaEventRecord(startEvent);
    gpu_copy<<<gridSize, blockSize>>>(d_A, d_B, total);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpu_copy_ms;
    cudaEventElapsedTime(&gpu_copy_ms, startEvent, stopEvent);

    std::cout << "GPU Copy Time: " << gpu_copy_ms / 1000.0f << " s\n";

    // ================= GPU TRANSPOSE =================
    cudaEventRecord(startEvent);
    gpu_transpose<<<gridSize, blockSize>>>(d_A, d_B, N);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpu_transpose_ms;
    cudaEventElapsedTime(&gpu_transpose_ms, startEvent, stopEvent);

    std::cout << "GPU Transpose Time: " << gpu_transpose_ms / 1000.0f << " s\n";

    // ================= BANDWIDTH =================
    double copy_bw = (2.0 * bytes) / (gpu_copy_ms / 1000.0) / 1e9;
    double transpose_bw = (2.0 * bytes) / (gpu_transpose_ms / 1000.0) / 1e9;

    std::cout << "GPU Copy Bandwidth: " << copy_bw << " GB/s\n";
    std::cout << "GPU Transpose Bandwidth: " << transpose_bw << " GB/s\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}