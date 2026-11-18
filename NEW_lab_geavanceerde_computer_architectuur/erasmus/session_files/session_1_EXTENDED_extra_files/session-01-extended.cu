// GCA
// session_01 extended
// joris.putteneers@kuleuven.be

// ================================================================

// session 1 extended — memory, synchronization and timing
//
// Builds on session 1 (threads, blocks, indexing, flipping an array) and adds more control:
// - memory: who moves the data between CPU and GPU, and when (implicit vs explicit)
// - waiting: different ways to wait for the GPU
// - timing: measuring each step, and expressing it as a speed (GB/s)
//
// Contents:
// R1   2D flip with implicit memory (cudaMallocManaged): hidden data movement
// R2   2D flip with explicit memory (cudaMalloc + cudaMemcpy): every phase timed, pinned memory
// R3   ways to wait for the GPU, a wrong timing, error checking, debug prints
// R4   four ways to flip a 2D array, timed: why the transpose is slower
// R5   intermediate exercise (solution): CPU vs GPU time as a function of array size
// plot.py  plots the R5 results

// ================================================================

// 08_flip_2d.cu  —  SOLUTION of the session-1 exercise

// Problem: a 2D array of W columns by H rows (think: an image). Reverse every row:
//   in:   0  1  2  3  4  5          out:  5  4  3  2  1  0
//         6  7  8  9 10 11                11 10  9  8  7  6
// Every thread handles one element and gets an x (column) and a y (row).
//
// Key idea: in memory the 2D array is still ONE long 1D array, stored row after row ("row-major"):
//   row 0 = positions 0..5, row 1 = positions 6..11, ...
//   so element (x, y) is at position  y * W + x.
// Mirroring a row: column x goes to column W - 1 - x (the 1D flip "n - 1 - i", applied per row),
//   so the destination is  y * W + (W - 1 - x).

#include <cstdio>

#define W 6
#define H 5

__global__ void flip2d(const int* in, int* out, int w, int h) {
    // Same formula as before, once for x (columns) and once for y (rows).
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Bounds check in both directions: there are more threads than columns AND more than rows.
    if (x < w && y < h) out[y * w + (w - 1 - x)] = in[y * w + x];
}

int main() {
    int *in, *out;
    cudaMallocManaged(&in, W * H * sizeof(int));
    cudaMallocManaged(&out, W * H * sizeof(int));

    for (int i = 0; i < W * H; i++) in[i] = i;

    // Blocks of 4 x 4 threads. The number of blocks is rounded up separately per direction:
    //   x: (6 + 3) / 4 = 2 blocks -> 8 threads for 6 columns (2 extra)
    //   y: (5 + 3) / 4 = 2 blocks -> 8 threads for 5 rows    (3 extra)
    // So 2 x 2 = 4 blocks of 16 threads = 64 threads for 30 elements; the bounds check stops the rest.
    // (Real code: e.g. dim3 threads(16, 16) = 256 threads per block.)
    dim3 threads(4, 4);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);
    flip2d<<<blocks, threads>>>(in, out, W, H);
    cudaDeviceSynchronize();

    printf("in:\n");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) printf("%3d", in[y * W + x]);
        printf("\n");
    }
    printf("out:\n");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) printf("%3d", out[y * W + x]);
        printf("\n");
    }

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (out[y * W + x] != in[y * W + (W - 1 - x)]) {
                printf("error at x=%d y=%d\n", x, y);
                return 1;
            }
        }
    }
    printf("correct\n");

    cudaFree(in);
    cudaFree(out);
}

// Output:
// in:
//   0  1  2  3  4  5
//   6  7  8  9 10 11
//  12 13 14 15 16 17
//  18 19 20 21 22 23
//  24 25 26 27 28 29
// out:
//   5  4  3  2  1  0
//  11 10  9  8  7  6
//  17 16 15 14 13 12
//  23 22 21 20 19 18
//  29 28 27 26 25 24
// correct

// Variations (same kernel, only the destination changes):
//   vertical flip (upside down):  out[(h - 1 - y) * w + x]           = in[y * w + x];
//   both (rotate 180 degrees):    out[(h - 1 - y) * w + (w - 1 - x)] = in[y * w + x];

// ================================================================

// exercises — session 1 extended
//
// Part A — implicit memory (R1_flip2d_managed.cu)
//   Run it.
//   - Why is run 1 slower than run 2, while the kernel is identical?
//   - Why is run 3 slower again?
//   - What does the GB/s value tell you?
//
// Part B — explicit memory (R2_flip2d_explicit.cu)
//   Run it.
//   - Which phase takes the longest: copy to GPU, kernel, or copy back?
//   - Compare the GB/s of the copies with the GB/s of the kernel. What limits the speed?
//   - Is the GPU total faster than the CPU flip?
//   - Set PINNED to 1. What changes, and why?
//
// Part C — synchronization and debugging (R3_sync.cu)
//   Run it.
//   - Which timing is wrong, and why?
//   - Why is D larger than C?
//   - What does E show? What would happen without cudaGetLastError()?
//   - Set DEBUG to 1 and check the index mapping of the first threads.
//
// Part D — other flips (R4_flip_variants.cu)
//   Run it.
//   - All four move the same amount of data. Which one is slowest, and why?
//
// Part E — intermediate exercise: CPU vs GPU as a function of array size
//   Write a program for the 1D flip that, for n = 2^10, 2^11, ... 2^25:
//   - times the CPU flip, the copy to the GPU, the kernel and the copy back separately
//   - checks every GPU result against the CPU result
//   - prints one CSV line per size: n,cpu_ms,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms
//   Plot the result with plot.py:
//     ./R5 > results.csv
//     python3 -m venv .venv
//     .venv/bin/pip install matplotlib
//     .venv/bin/python plot.py results.csv        -> flip_timing.png
//   Questions:
//   - From which n is the kernel alone faster than the CPU?
//   - Is the GPU including the copies ever faster than the CPU?
//   - What dominates the GPU time for large n?
//   - For what kind of problem is a GPU worth it, and for what kind not?

// ================================================================

// R1_flip2d_managed.cu  —  2D flip with Unified Memory (IMPLICIT memory management)

// A Full HD "image" (1920 x 1080 ints = 8.3 MB) is flipped horizontally, exactly like exercise 08.
// One pointer per array: the driver moves the data between CPU and GPU by itself, when it is needed.
// We cannot see or control WHEN that happens, but we can see its effect in the timing:
//   run 1: the data still lives in CPU memory -> it is moved to the GPU during this run -> slow
//   run 2: the data is already on the GPU                                              -> fast
//   run 3: the CPU just read the result (verification) -> "out" moved back to the CPU,
//          so it has to travel to the GPU again                                       -> slower again
// The kernel is identical in all three runs; only the hidden data movement differs.
//
// Bandwidth ("bitrate"): how many bytes per second are moved. The flip reads every element once
// and writes it once, so it moves 2 * 8.3 MB per run. GB/s = bytes / time.
// (1 GB/s = 8 Gbit/s.) For reference:
//   GPU memory (on the card)     : hundreds of GB/s on modern GPUs
//   PCIe bus (between CPU & GPU) : roughly 8 to 64 GB/s, depending on the PCIe generation
// So once data has to cross the bus, the bus is the bottleneck, not the GPU.
//
// Lab PCs (GT 730): compile with  nvcc -arch=sm_35 -o R1 R1_flip2d_managed.cu

#include <cstdio>

#define W 1920
#define H 1080

__global__ void flip2d(const int* in, int* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y * w + (w - 1 - x)] = in[y * w + x];
}

float run(const int* in, int* out, dim3 blocks, dim3 threads) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    flip2d<<<blocks, threads>>>(in, out, W, H);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    size_t bytes = (size_t)W * H * sizeof(int);
    int *in, *out;
    cudaMallocManaged(&in, bytes);
    cudaMallocManaged(&out, bytes);

    for (int i = 0; i < W * H; i++) in[i] = i;

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    float ms1 = run(in, out, blocks, threads);
    float ms2 = run(in, out, blocks, threads);

    int errors = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (out[y * W + (W - 1 - x)] != in[y * W + x]) errors++;

    float ms3 = run(in, out, blocks, threads);

    printf("run 1 (data still on CPU)   : %8.3f ms  %7.1f GB/s\n", ms1, 2.0 * bytes / (ms1 * 1e6));
    printf("run 2 (data already on GPU) : %8.3f ms  %7.1f GB/s\n", ms2, 2.0 * bytes / (ms2 * 1e6));
    printf("run 3 (after CPU read out)  : %8.3f ms  %7.1f GB/s\n", ms3, 2.0 * bytes / (ms3 * 1e6));
    printf("%s\n", errors == 0 ? "correct" : "WRONG");

    cudaFree(in);
    cudaFree(out);
}

// Output (times depend on the machine; the pattern is what matters: run 1 and 3 slower than run 2):
// run 1 (data still on CPU)   :    ... ms     ... GB/s
// run 2 (data already on GPU) :    ... ms     ... GB/s
// run 3 (after CPU read out)  :    ... ms     ... GB/s
// correct

// ================================================================

// R2_flip2d_explicit.cu  —  the same 2D flip with EXPLICIT memory management

// Now there are two copies of every array:
//   h_in, h_out : in CPU memory ("host")
//   d_in, d_out : in GPU memory ("device")
// and WE decide when data crosses the bus:
//   1. cudaMemcpy(d_in, h_in, ..., cudaMemcpyHostToDevice)    CPU -> GPU
//   2. kernel works on d_in / d_out                           only GPU memory
//   3. cudaMemcpy(h_out, d_out, ..., cudaMemcpyDeviceToHost)  GPU -> CPU
// Every phase is timed separately with CUDA events, so the cost of moving data becomes visible.
//
// What to look for:
//   - the kernel reaches a much higher GB/s than the copies: GPU memory is far faster than the bus
//   - for a simple kernel like the flip, the copies take MORE time than the kernel itself
//   - so: move data as little as possible, and keep it on the GPU as long as possible
//
// Setting to try: PINNED
//   0: host arrays from malloc ("pageable" memory). The driver first copies them into an internal
//      buffer that the GPU can access, then over the bus: an extra step.
//   1: host arrays from cudaMallocHost ("pinned" memory): the GPU reads/writes them directly over
//      the bus, usually noticeably faster. Pinned memory is a limited resource (it locks RAM),
//      so only use it for buffers that are transferred often. Also required for async copies (session 5).
//
// Lab PCs (GT 730): compile with  nvcc -arch=sm_35 -o R2 R2_flip2d_explicit.cu

#include <cstdio>
#include <cstdlib>
#include <chrono>

#define W 1920
#define H 1080
#define PINNED 0

__global__ void flip2d(const int* in, int* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y * w + (w - 1 - x)] = in[y * w + x];
}

void flip2d_cpu(const int* in, int* out, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            out[y * w + (w - 1 - x)] = in[y * w + x];
}

double gbs(double bytes, double ms) {
    return bytes / (ms * 1e6);
}

int main() {
    size_t bytes = (size_t)W * H * sizeof(int);

    int *h_in, *h_out;
    if (PINNED) {
        cudaMallocHost(&h_in, bytes);
        cudaMallocHost(&h_out, bytes);
    } else {
        h_in = (int*)malloc(bytes);
        h_out = (int*)malloc(bytes);
    }
    int* h_ref = (int*)malloc(bytes);
    for (int i = 0; i < W * H; i++) h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    auto c0 = std::chrono::steady_clock::now();
    flip2d_cpu(h_in, h_ref, W, H);
    auto c1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    flip2d<<<blocks, threads>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);

    cudaEventRecord(e0);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);
    flip2d<<<blocks, threads>>>(d_in, d_out, W, H);
    cudaEventRecord(e2);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    float h2d, kernel, d2h;
    cudaEventElapsedTime(&h2d, e0, e1);
    cudaEventElapsedTime(&kernel, e1, e2);
    cudaEventElapsedTime(&d2h, e2, e3);

    int errors = 0;
    for (int i = 0; i < W * H; i++)
        if (h_out[i] != h_ref[i]) errors++;

    printf("pinned host memory: %s\n", PINNED ? "yes" : "no");
    printf("CPU flip          : %8.3f ms  %7.1f GB/s\n", cpu_ms, gbs(2.0 * bytes, cpu_ms));
    printf("copy CPU -> GPU   : %8.3f ms  %7.1f GB/s\n", h2d, gbs(bytes, h2d));
    printf("kernel            : %8.3f ms  %7.1f GB/s\n", kernel, gbs(2.0 * bytes, kernel));
    printf("copy GPU -> CPU   : %8.3f ms  %7.1f GB/s\n", d2h, gbs(bytes, d2h));
    printf("GPU total         : %8.3f ms\n", h2d + kernel + d2h);
    printf("%s\n", errors == 0 ? "correct" : "WRONG");

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaFree(d_in);
    cudaFree(d_out);
    if (PINNED) {
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    } else {
        free(h_in);
        free(h_out);
    }
    free(h_ref);
}

// Output (times depend on the machine):
// pinned host memory: no
// CPU flip          :    ... ms     ... GB/s
// copy CPU -> GPU   :    ... ms     ... GB/s
// kernel            :    ... ms     ... GB/s
// copy GPU -> CPU   :    ... ms     ... GB/s
// GPU total         :    ... ms
// correct

// ================================================================

// R3_sync.cu  —  four ways to wait for the GPU (and one way to time it wrong), plus error checking

// The same 2D flip (explicit memory), launched several times. Each time we wait in a different way:
//   A  CPU timer around the launch only                 -> WRONG: measures only how long it takes
//                                                          to START the kernel (launch is asynchronous)
//   B  CPU timer + cudaDeviceSynchronize()               -> kernel time + a bit of CPU overhead.
//                                                          Waits for ALL GPU work: simple, the "big hammer"
//   C  CUDA events + cudaEventSynchronize(stop)          -> kernel time measured by the GPU itself:
//                                                          the most precise. Waits only until "stop"
//   D  CPU timer + cudaMemcpy back (no explicit wait)    -> kernel + copy: cudaMemcpy waits for the kernel
//                                                          by itself, then copies, then returns
//   E  launch with too many threads per block            -> the kernel never runs; only cudaGetLastError()
//                                                          tells you
//
// When does the choice matter?
//   - one kernel after another (like here): B, C and D are all correct, use what is convenient
//   - timing: use events (C); a CPU timer needs a wait (B), otherwise you get A
//   - several streams / CPU work in parallel (session 5): cudaDeviceSynchronize() makes everything
//     wait for everything; waiting on a specific event keeps the rest running
//   - debugging: synchronize + check errors after every kernel, so an error shows up next to its cause
//
// DEBUG = 1 prints what the first few threads do. Print only a few threads: 2 million threads that
// all print would flood the output.
// Memory errors (writing outside an array) are not reported by printf; run:  compute-sanitizer ./R3
//
// Lab PCs (GT 730): compile with  nvcc -arch=sm_35 -o R3 R3_sync.cu

#include <cstdio>
#include <cstdlib>
#include <chrono>

#define W 1920
#define H 1080
#define DEBUG 0

__global__ void flip2d(const int* in, int* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int src = y * w + x;
        int dst = y * w + (w - 1 - x);
        if (DEBUG && x < 2 && y < 2) printf("x=%d y=%d: in[%d] -> out[%d]\n", x, y, src, dst);
        out[dst] = in[src];
    }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    size_t bytes = (size_t)W * H * sizeof(int);
    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);
    for (int i = 0; i < W * H; i++) h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    flip2d<<<blocks, threads>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::steady_clock::now();
    flip2d<<<blocks, threads>>>(d_in, d_out, W, H);
    double a = ms_since(t0);
    cudaDeviceSynchronize();

    t0 = std::chrono::steady_clock::now();
    flip2d<<<blocks, threads>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    double b = ms_since(t0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    flip2d<<<blocks, threads>>>(d_in, d_out, W, H);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float c;
    cudaEventElapsedTime(&c, start, stop);

    t0 = std::chrono::steady_clock::now();
    flip2d<<<blocks, threads>>>(d_in, d_out, W, H);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    double d = ms_since(t0);

    int errors = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (h_out[y * W + (W - 1 - x)] != h_in[y * W + x]) errors++;

    printf("A  CPU timer, no wait (WRONG)          : %8.3f ms\n", a);
    printf("B  CPU timer + cudaDeviceSynchronize   : %8.3f ms\n", b);
    printf("C  events + cudaEventSynchronize       : %8.3f ms\n", c);
    printf("D  CPU timer + cudaMemcpy (waits itself): %8.3f ms  (kernel + copy back)\n", d);
    printf("result: %s\n", errors == 0 ? "correct" : "WRONG");

    flip2d<<<blocks, dim3(64, 32)>>>(d_in, d_out, W, H);
    cudaError_t err = cudaGetLastError();
    printf("E  launch with 64 x 32 = 2048 threads per block: %s\n", cudaGetErrorString(err));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}

// Output (times depend on the machine; A is far smaller than the rest, C is the real kernel time):
// A  CPU timer, no wait (WRONG)          :    ... ms
// B  CPU timer + cudaDeviceSynchronize   :    ... ms
// C  events + cudaEventSynchronize       :    ... ms
// D  CPU timer + cudaMemcpy (waits itself):    ... ms  (kernel + copy back)
// result: correct
// E  launch with 64 x 32 = 2048 threads per block: invalid configuration argument

// ================================================================

// R4_flip_variants.cu  —  four ways to flip a 2D array, timed and verified

//   horizontal : mirror every row                     (exercise 08)
//   vertical   : upside down
//   rotate 180 : both
//   transpose  : flip over the diagonal (row y becomes column y)
// All four read and write exactly the same amount of data, so you would expect the same time.
// They are not the same: the transpose is clearly slower.
// Why: in the first three, threads that are neighbours (x, x+1) also write to neighbouring addresses,
// so the GPU combines their writes into a few large memory transactions.
// In the transpose, neighbouring threads write addresses that are a whole row apart (h ints),
// so every write becomes its own small transaction. HOW you access memory matters, not only how much.
// (This is the topic of the Memory session: "coalesced" memory access.)
//
// New: __host__ __device__ — dest() is compiled for BOTH the CPU and the GPU,
// so the kernel and the CPU verification use exactly the same index formula.
//
// Lab PCs (GT 730): compile with  nvcc -arch=sm_35 -o R4 R4_flip_variants.cu

#include <cstdio>
#include <cstdlib>

#define W 2048
#define H 2048

__host__ __device__ int dest(int x, int y, int w, int h, int mode) {
    if (mode == 0) return y * w + (w - 1 - x);
    if (mode == 1) return (h - 1 - y) * w + x;
    if (mode == 2) return (h - 1 - y) * w + (w - 1 - x);
    return x * h + y;
}

__global__ void flip(const int* in, int* out, int w, int h, int mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[dest(x, y, w, h, mode)] = in[y * w + x];
}

int main() {
    const char* names[] = {"horizontal", "vertical", "rotate 180", "transpose"};
    size_t bytes = (size_t)W * H * sizeof(int);

    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);
    for (int i = 0; i < W * H; i++) h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int mode = 0; mode < 4; mode++) {
        flip<<<blocks, threads>>>(d_in, d_out, W, H, mode);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        flip<<<blocks, threads>>>(d_in, d_out, W, H, mode);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (h_out[dest(x, y, W, H, mode)] != h_in[y * W + x]) errors++;

        printf("%-11s %8.3f ms  %7.1f GB/s  %s\n", names[mode], ms, 2.0 * bytes / (ms * 1e6),
               errors == 0 ? "correct" : "WRONG");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}

// Output (times depend on the machine; transpose is the slowest):
// horizontal       ... ms      ... GB/s  correct
// vertical         ... ms      ... GB/s  correct
// rotate 180       ... ms      ... GB/s  correct
// transpose        ... ms      ... GB/s  correct

// ================================================================

// R5_benchmark.cu  —  INTERMEDIATE EXERCISE (solution): CPU vs GPU time as a function of array size

// The 1D flip (example 06), with explicit memory, for n = 1024 ... 33 million elements.
// For every size it measures:
//   cpu_ms        the flip on the CPU
//   h2d_ms        copy CPU -> GPU
//   kernel_ms     the kernel only
//   d2h_ms        copy GPU -> CPU
//   gpu_total_ms  everything the GPU version needs: copy + kernel + copy back
// and checks every GPU result against the CPU result.
// The output is CSV (comma separated values): one line per size, ready to plot with plot.py.
//
// Run:
//   nvcc -o R5 R5_benchmark.cu            (lab PCs: add -arch=sm_35)
//   ./R5 > results.csv
//
// What the plot typically shows (verify on your own machine):
//   - small arrays: the CPU wins. A kernel launch + copies cost a fixed few microseconds.
//   - kernel only: overtakes the CPU once the array is large enough.
//   - GPU total: the copies dominate. The flip does almost no computation per byte, so moving the
//     data over the bus costs more than flipping it on the CPU. The GPU only pays off when there is
//     enough computation per byte, or when the data stays on the GPU for several kernels.

#include <cstdio>
#include <cstdlib>
#include <chrono>

__global__ void flip(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[n - 1 - i] = in[i];
}

void flip_cpu(const int* in, int* out, int n) {
    for (int i = 0; i < n; i++) out[n - 1 - i] = in[i];
}

int main() {
    const int max_n = 1 << 25;
    size_t max_bytes = (size_t)max_n * sizeof(int);

    int* h_in = (int*)malloc(max_bytes);
    int* h_out = (int*)malloc(max_bytes);
    int* h_ref = (int*)malloc(max_bytes);
    for (int i = 0; i < max_n; i++) {
        h_in[i] = i;
        h_out[i] = 0;
        h_ref[i] = 0;
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, max_bytes);
    cudaMalloc(&d_out, max_bytes);

    flip<<<1, 32>>>(d_in, d_out, 32);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);

    printf("n,cpu_ms,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms\n");
    for (int n = 1 << 10; n <= max_n; n *= 2) {
        size_t bytes = (size_t)n * sizeof(int);

        auto c0 = std::chrono::steady_clock::now();
        flip_cpu(h_in, h_ref, n);
        auto c1 = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        cudaEventRecord(e0);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(e1);
        flip<<<blocks, threads>>>(d_in, d_out, n);
        cudaEventRecord(e2);
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);

        float h2d, kernel, d2h;
        cudaEventElapsedTime(&h2d, e0, e1);
        cudaEventElapsedTime(&kernel, e1, e2);
        cudaEventElapsedTime(&d2h, e2, e3);

        for (int i = 0; i < n; i++) {
            if (h_out[i] != h_ref[i]) {
                fprintf(stderr, "error at n=%d, i=%d\n", n, i);
                return 1;
            }
        }

        printf("%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", n, cpu_ms, h2d, kernel, d2h, h2d + kernel + d2h);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);
}

// fprintf(stderr, ...) prints errors to the terminal, not into results.csv.
//
// Output (results.csv; the numbers depend on the machine):
// n,cpu_ms,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms
// 1024,...,...,...,...,...
// 2048,...,...,...,...,...
// ...
// 33554432,...,...,...,...,...

// ================================================================

// # plot.py  (Python, save as plot.py)
//
// import csv
// import sys
//
// import matplotlib.pyplot as plt
//
// path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
// with open(path) as f:
//     rows = list(csv.DictReader(f))
//
// n = [int(r["n"]) for r in rows]
// series = [
//     ("cpu_ms", "CPU"),
//     ("kernel_ms", "GPU kernel only"),
//     ("gpu_total_ms", "GPU incl. copies"),
//     ("h2d_ms", "copy CPU -> GPU"),
//     ("d2h_ms", "copy GPU -> CPU"),
// ]
// for key, label in series:
//     style = "--" if key.endswith("2d_ms") else "-"
//     plt.plot(n, [float(r[key]) for r in rows], style, marker="o", markersize=3, label=label)
//
// plt.xscale("log", base=2)
// plt.yscale("log")
// plt.xlabel("array length n (elements)")
// plt.ylabel("time (ms)")
// plt.title("Array flip: CPU vs GPU")
// plt.grid(True, which="both", alpha=0.3)
// plt.legend()
// plt.tight_layout()
// plt.savefig("flip_timing.png", dpi=150)
// print("saved flip_timing.png")

// ================================================================

// TEACHER ONLY (remove before handing out) — expected observations
//
// Part A (implicit):
// - run 1 includes moving in/out to the GPU: the transfer is hidden inside the kernel time
// - run 2: data already on the GPU -> the real kernel speed
// - run 3: the CPU read "out" during verification, so it moved back; it has to travel again
// - lesson: with managed memory the data movement is invisible but not free;
//   an innocent CPU access can make data travel back and forth
//
// Part B (explicit):
// - the copies are much slower (in GB/s) than the kernel: the bus is the bottleneck
// - for the flip, copy time > kernel time; GPU total is often not faster than the CPU
// - PINNED = 1: copies faster (no extra internal copy step); costs locked RAM
//
// Part C (synchronization):
// - A is tiny: only the launch, the kernel is still running when the timer stops
// - B ~ C (+ a little CPU overhead); C is the most precise (measured by the GPU)
// - D = kernel + copy back, because cudaMemcpy waits for the kernel by itself
// - E: "invalid configuration argument". Without the check: nothing happens, no message,
//   the output array simply keeps its old content
//
// Part D (variants):
// - horizontal / vertical / rotate 180: about the same time
// - transpose: slower. Neighbouring threads write addresses a whole row apart, so the writes
//   cannot be combined (no coalescing). Preview of the Memory session.
//
// Part E (benchmark):
// - small n: CPU fastest (fixed launch + copy overhead of microseconds)
// - kernel only: beats the CPU for large n
// - GPU total: dominated by the copies; for the flip it may never beat the CPU
// - a GPU pays off with much computation per byte moved, or when data stays on the GPU
//   for several kernels. The flip is the opposite: no computation, only data movement.
