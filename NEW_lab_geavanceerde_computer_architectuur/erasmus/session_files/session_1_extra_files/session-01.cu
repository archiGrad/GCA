// GCA
// session_01
// 12-11-2026
// joris.putteneers@kuleuven.be

// ================================================================

// Reading
//
// Read the first chapter of the CUDA Programming Guide:
// https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html
//
// With focus on:
// - CUDA and the CUDA Programming Guide
// - 1.1.1. The Graphics Processing Unit
// - 1.1.2. The Benefits of Using GPUs
// - 1.1.3. Getting Started Quickly
// - 1.2.2.1. Thread Blocks and Grids
// - 1.2.2.1.1. Thread Block Clusters
// - 1.2.2.2. Warps and SIMT
// - 1.2.3.4. Unified Memory



// ================================================================


// In the following exercises, every new piece of functionality has a comment above it that explains it.

// ================================================================

// Examples: "hello" to array flip

// ================================================================

// 01_hello.cu
//
// description: "Write a .cu file that prints a hello message once from the GPU and once from the CPU."
//
//
// short version
//
#include <cstdio>
__global__ void hello(){
	printf("hello from gpu\n");
}
int main(){
	hello<<<1,1>>>();
	cudaDeviceSynchronize();
	printf("hi from cpu\n");
}

//long version

// <cstdio> is the C standard input/output library (C++ name for stdio.h).
#include <cstdio>

// __global__ marks a kernel: it is CALLED from the CPU but RUNS on the GPU.
// "global" because it is visible to both sides. The only way to run your own code on the GPU is through a kernel. Helpers marked __device__ can then be called from a kernel.
// void is the RETURN TYPE. __global__ is an "execution space specifier"
__global__ void hello() {
    // this printf runs on the GPU and is NOT the stdio one. It has the same name and the buffer in GPU memory.
    printf("Hello from the GPU\n");
}

// main() ALWAYS runs on the CPU. Only the body of hello() runs on the GPU.
// In C++ main must return int; e.g. "void main()" is a compile error ('main' must return 'int').
//  main is the only function where you may leave out "return 0;": it then returns 0 automatically. That is why this code has no return.
int main() {
    // Launch the kernel on the GPU: <<<arg 1 =number of blocks, arg 2 = threads per block>>>.
    // " launch the hello kernel with 1 block of 1 thread".
    // Each thread runs the whole body of hello() once, so 1 thread = 1 line of output.
    // hello<<<2, 8>>>() would be 2 blocks of 8 threads = 16 threads which would be 16 times "Hello from the GPU".
    // The launch is asynchronous: the CPU does not wait and continues immediately.
    hello<<<1, 1>>>();

    // cudaDeviceSynchronize() is a CUDA function, called on the CPU, that makes the CPU  wait until the GPU has finished all launched work. It also flushes the GPU's printf output. Then the CPU simply continues with the next line, as it always would. Without it, the program could end before the GPU has printed anything.
    // cudaDeviceSynchronize = wait for the GPU
    cudaDeviceSynchronize();


    // this is the normal C printf from <cstdio>, running on the CPU and writing to the terminal.
    printf("Hello from the CPU\n");
}

// Output:
// Hello from the GPU
// Hello from the CPU
//




// ================================================================

// 02_threads.cu
//
//description: Launch 1 block of 8 threads; each thread prints its own index (threadIdx.x) on a new line.


// short version 
#include <cstdio>

__global__ void hello(){
	printf("current thread %d\n", threadIdx.x);

}
int main(){
	hello<<<1, 8>>>();
	cudaDeviceSynchronize();
}

//long version

// even though no printf statement is made, it is best practice to include #include <cstdio> . nvcc still compiles without it, because nvcc automatically includes cuda_runtime.h, which includes stdio.h. 
#include <cstdio>

__global__ void hello() {
    // threadIdx.x is a built-in variable that exists automatically in every kernel.
    // It is the index of this thread within its block: 0..7 here.
    // Every thread sees a different value; that is how threads tell themselves apart.
    //
    // threads each run this line once, so you get 8 lines, 0 to 7.
    //
    // %d is a placeholder: printf replaces it with the integer argument (threadIdx.x).
    // \n is a newline.
    // (std::cout does not work on the GPU, only printf.)
    //
    // threadIdx is ONE variable, a struct with three fields: threadIdx.x, threadIdx.y, threadIdx.z.
    // (There is no "threadIdy"; it is always threadIdx.something.)
    // With a 1D launch like <<<1, 8>>>, only .x changes; .y and .z are always 0.
    printf("thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 8>>>();
    cudaDeviceSynchronize();
}

// Output (order is not guaranteed, but within one block it is usually in order):
// thread 0
// thread 1
// thread 2
// thread 3
// thread 4
// thread 5
// thread 6
// thread 7
//
// Why not guaranteed: all threads run in parallel, and the GPU's hardware scheduler decides which threads execute when. Threads are run in groups of 32 (warps). Within one warp the printf usually comes out in thread order, but CUDA does not promise it. Between warps and between blocks there is no order at all: blocks can run on different parts of the GPU, at the same time, in any order.  never write code that depends on thread order!

// Figure: Threads are organized into a grid of blocks.  (figures/session_1_figures/slide-15.pdf)




// ================================================================

// 02b_threads_2d.cu  (2D thread block" or "multidimensional thread indexing".)
//

//description: Launch 1 block of 4 x 2 threads using dim3; each thread prints its threadIdx.x and threadIdx.y."
//short version
 #include <cstdio>

__global__ void hello(){
	printf("current threadIdx.x x=%d and threadIdx.y y= %d\n", threadIdx.x, threadIdx.y);
}
int main(){
	dim3 threads(4, 2);
	hello<<<1, threads >>>();
	cudaDeviceSynchronize();
}

// long version
//
//  dim3 is a CUDA built-in vector type (a struct with x, y, z).
//
// Why threadIdx.x or threadIdx.y or threadIdx.z... 2D/3D? It is purely a convenience for the programmer, not a hardware feature.
// The hardware still sees the threads as one long line (x counts fastest, then y, then z)
// and groups them into warps of 32 in that order. Performance is the same.
// The benefit: the indices match the shape of the data.
//   - 2D: images (x = column, y = row), matrices.  3D: volumes (CT scans, voxels, 3D simulations).
//   - Without it you would compute row = i / W and col = i % W yourself; with 2D you get them directly.
// Very common: almost every image or matrix kernel uses 2D blocks, e.g. 16 x 16 threads.
// Every CUDA GPU supports 1D, 2D and 3D blocks and grids (block z is limited to 64).
// Also: x * y * z <= 1024 threads per block in total. dim3(32, 32) = 1024 is the max for a 2D block.
#include <cstdio>

__global__ void hello() {
    //  threadIdx and blockIdx are of type uint3, blockDim and gridDim are of type dim3.]
    // Now the block is 2D, so every thread has two indices:
    //   threadIdx.x = column inside the block: 0..3
    //   threadIdx.y = row inside the block:    0..1
    printf("thread x=%d y=%d\n", threadIdx.x, threadIdx.y);
}

int main() {
    // dim3 is a CUDA type with three numbers (x, y, z); missing ones default to 1.
    // dim3 threads(4, 2) = a block of 4 columns by 2 rows = 8 threads.
    // Writing <<<1, 8>>> is shorthand for dim3(1, 1, 1) blocks and dim3(8, 1, 1) threads.
    // The grid can be 2D in the same way: dim3 blocks(2, 3) gives blockIdx.x and blockIdx.y.
    // 2D is convenient for images and matrices: x = column, y = row.
    // dim3 is a CUDA TYPE: a small struct with three numbers x, y, z (like int, but with three values).
    // "threads"is just the name of our variable;
    // The built-ins are only threadIdx, blockIdx, blockDim and gridDim. There is a 5th built-in: warpSize (= 32). which we cover later.
    dim3 threads(4, 2);
    hello<<<1, threads>>>();
    cudaDeviceSynchronize();
}

// Output (same 8 threads as 02, but now each has an x and a y; order not guaranteed):
// thread x=0 y=0
// thread x=1 y=0
// thread x=2 y=0
// thread x=3 y=0
// thread x=0 y=1
// thread x=1 y=1
// thread x=2 y=1
// thread x=3 y=1


this is a good ender of practice after flippoing 2d array too
i feel this example should be more explained. i still dont really understand what you cna do with this.. maybe 2 more  practical psuedocode example of this dim () would be good...    like a more practival example with an image or something liek 2d or 3d  array
[Two practical examples. They use the global index from 03 and the bounds check from 04.

 1) 2D: invert a grayscale image of W x H pixels.
    In memory the image is ONE long array, row after row: pixel (x, y) is at index y * W + x.

    __global__ void invert(unsigned char* img, int W, int H) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
        int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
        if (x < W && y < H) {
            int i = y * W + x;
            img[i] = 255 - img[i];
        }
    }

    dim3 threads(16, 16);                         // each block = a 16 x 16 tile of pixels
    dim3 blocks((W + 15) / 16, (H + 15) / 16);    // enough tiles to cover the image, rounded up
    invert<<<blocks, threads>>>(img, W, H);

    1920 x 1080 image: blocks = (120, 68) -> 1920 x 1088 threads, 1 thread per pixel.
    The 8 extra rows of threads are stopped by the bounds check.
    In 1D you would launch W * H threads and compute x = i % W, y = i / W yourself. Same result, less readable.

 2) 3D: threshold a volume (CT scan, voxel grid) of X x Y x Z voxels.
    Voxel (x, y, z) is at index (z * Y + y) * X + x.

    __global__ void threshold(float* vol, int X, int Y, int Z, float t) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x < X && y < Y && z < Z) {
            int i = (z * Y + y) * X + x;
            vol[i] = vol[i] > t ? 1.0f : 0.0f;
        }
    }

    dim3 threads(8, 8, 8);                                  // each block = an 8 x 8 x 8 cube of voxels (512 threads)
    dim3 blocks((X + 7) / 8, (Y + 7) / 8, (Z + 7) / 8);
    threshold<<<blocks, threads>>>(vol, X, Y, Z, 0.5f);
]




// ================================================================

// 03_blocks.cu
//
//  description: Launch 2 blocks of 4 threads; each thread prints its unique global index.

#include <cstdio>

__global__ void hello() {
    // besides threadIdx , there are lso built-in variables:
    // blockIdx.x = index of this block in the grid: 0..1 here    <-- also struct?
    // blockDim.x = threads per block, i.e. the 2nd launch number: 4 here   <-also struct?
    // gridDim.x  = number of blocks, i.e. the 1st launch number: 2 here  <-- also struct?
    // [Yes, all three. blockIdx is a uint3, blockDim and gridDim are dim3: structs with .x .y .z, same as threadIdx.]
    // [typo: "lso" -> "also"]
    //
    // to get a unique global index: block 1, thread 2 -> 1*4+2 = 6.
    // global index = blockIdx.x * blockDim.x + threadIdx.x
    //
    //exampple matrix where we do blockidx * threadixx  to see that this is not good enough, w
    //
    //0x0 = 0
    //0x1 = ...
    //0x2
    //0x3
    //1x0
    //1x1
    //1x2
    //1x3
    //1x4
    //
    //but if you blockix x * blockdimx + threadidx ,      ytou cna put this next to the other example  not below
    // 0x4=0
    //...
    // 1*4+4 = 7
    //
    // [Side-by-side version of your draft above. Your draft had "1x4" and "1*4+4": those don't exist, threadIdx.x only goes 0..3.
    //
    //  block  thread | wrong: blockIdx.x * threadIdx.x | right: blockIdx.x * blockDim.x + threadIdx.x
    //    0      0    |   0 * 0 = 0                     |   0 * 4 + 0 = 0
    //    0      1    |   0 * 1 = 0                     |   0 * 4 + 1 = 1
    //    0      2    |   0 * 2 = 0                     |   0 * 4 + 2 = 2
    //    0      3    |   0 * 3 = 0                     |   0 * 4 + 3 = 3
    //    1      0    |   1 * 0 = 0                     |   1 * 4 + 0 = 4
    //    1      1    |   1 * 1 = 1                     |   1 * 4 + 1 = 5
    //    1      2    |   1 * 2 = 2                     |   1 * 4 + 2 = 6
    //    1      3    |   1 * 3 = 3                     |   1 * 4 + 3 = 7
    //
    //  wrong: 0 appears 5 times and 4..7 never: threads overwrite each other's element, half the array is never touched.
    //  right: every thread gets a different index, 0..7, no gaps.]
    //
    int i = blockIdx.x * blockDim.x + threadIdx.x;      <-- fixed these
    // [Correct.]
    printf("global index %d\n", i);
}

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
}

// Output (block 1 may print before block 0; that differs between runs):
// [fixed output:]
// global index 0
// global index 1
// global index 2
// global index 3
// global index 4
// global index 5
// global index 6
// global index 7

// Figure: You decide how to divide the grid.  (figures/session_1_figures/slide-17.pdf)

// Figure: Correct indexing: add the block offset to get the absolute position.  (figures/session_1_figures/slide-20.pdf)



short version <---
[MISTAKES (does not compile):
 - "__global__void": needs a space -> __global__ void
 - blockIdx * blockDim + threadIdx: needs .x on all three (they are structs; structs cannot be multiplied)
 - missing ; at the end of that line
 - "%d\n ": the space after \n puts a space at the start of the next line; remove it
 - #include <cstdio> missing (compiles with nvcc anyway)]

__global__void hello(){
	int i = blockIdx * blockDim + threadIdx
	printf( "individual global index = %d\n ",i );
}


int main(){
	hello<<<2, 4>>>();
	cudaDeviceSynchronize();
}


// ================================================================

// 04_memory.cu
//
// description:  Allocate an array in unified memory, let each GPU thread write its global index into its own element, then print the array on the CPU.

//short version

#include <cstdio>
#define N 8


__global__ void fill(int* a){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	a[i] = i;
}

int main(){
	int* a;
	cudaMallocManaged(&a, N * sizeof(int));
	fill<<<2,4>>>(a);
	cudaDeviceSynchronize();
	for(int i = 0; i < N; i++)printf("%d ", a[i]);
	cudaFree(a);
}

//long version
//  Every line in main() runs on the CPU; only the kernel fill() runs on the GPU.
//   1. int* a;                                   (CPU) Create an empty pointer. It points nowhere yet.
//   2. cudaMallocManaged(&a, N * sizeof(int));   (CPU calls it) Reserve 32 bytes that both CPU and GPU
//                                                can reach, and put the address into a. The memory
//                                                exists now, but holds no useful values yet.
//   3. fill<<<2, 4>>>(a);                        (CPU starts it, GPU runs it) 8 GPU threads each write
//                                                their own index into their own element: 0 1 2 ... 7.
//                                                The CPU does not wait here.
//   4. cudaDeviceSynchronize();                  (CPU) Wait until the GPU has finished filling.
//   5. for loop with printf                      (CPU) Read the array and print 0 1 2 3 4 5 6 7.
//   6. cudaFree(a);                              (CPU calls it) Give the memory back; we are done with it.
#include <cstdio>

// Preprocessor constant: before compiling, every N in the file is replaced by 8.
// It represents the length of the array.
#define N 8

// A kernel that receives a pointer (the address of the array).
// It does not MAKE the array: the memory is reserved in main() by cudaMallocManaged.
// This kernel FILLS the existing array: it writes the values 0..7 into it.
//
// This replaces the CPU loop "for (i = 0; i < N; i++) a[i] = i;" — the threads ARE the loop.
// int* = "pointer to int": a holds a memory ADDRESS, not a number. The kernel needs that address to write into  the array that main() allocated. Kernel arguments are copied, but a copied address still points to the same  memory, so the writes are visible in main(). A plain "int a" would only be a copy of one number.
__global__ void fill(int* a) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;  <-- why is there here no asterix?
    // No bounds check needed HERE because N = 8 and we launch exactly 2 * 4 = 8 threads:
    a[i] = i;
}

int main() {
    // a is a pointer: a variable that holds a memory address (the start of an int array).
    // It does not point anywhere yet.
    int* a;

    // Allocate N * sizeof(int) = 8 * 4 = 32 bytes of unified memory,  C++ itself does not guarantee 4 bytes, which is why you write sizeof(int) instead of 4.
    // reachable by both the CPU and the GPU.
    // &a is the address of the pointer itself, so the function can write the new address into a.  and is the & symbol refering to adress.
    // [Yes. & in front of a variable = "address of". a is an int*, so &a is an int** (address of the pointer).
    //  cudaMallocManaged needs it to change a itself.
    cudaMallocManaged(&a, N * sizeof(int));

    fill<<<2, 4>>>(a);

    // The CPU waits until the kernel is done. Only then is it safe to read a on the CPU.
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) printf("%d ", a[i]);
    printf("\n");

    // Release the memory from cudaMallocManaged (the CUDA equivalent of free()).
    cudaFree(a);
}

// Output:
// 0 1 2 3 4 5 6 7

// Why we need bounds:
// The number of threads you launch is always blocks x threads per block, so it's always a multiple of the block size. The size of your data (N) is set by the problem and almost never matches. The bounds check if (i < N) stops the extra threads from reading or writing memory outside the array. Without it, the program can silently corrupt memory or crash, even when the output looks correct.

// Figure: Which part of the program runs on the CPU and which on the GPU.  (figures/session_1_figures/slide-08.pdf)


// ================================================================

// 05_bounds.cu
//
//
// do the same exercise but amount of memory we need to make free is not equal to blocks times threads. is this correct?

#include <cstdio>

#define N 10

__global__ void fill(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // We launch 3 blocks * 4 threads = 12 threads, but the array has only 10 elements.
    // Threads 10 and 11 must do nothing, otherwise they write outside the array.
    if (i < n) a[i] = i;
    
}

int main() {
    int* a;
    cudaMallocManaged(&a, N * sizeof(int));

    int threads = 4;
    // Integer division rounds DOWN: 10 / 4 = 2 blocks = 8 threads, which is too few.
    // Adding (threads - 1) makes it round UP: (10 + 3) / 4 = 3 blocks = 12 threads.
    int blocks = (N + threads - 1) / threads;

    // N is the #define above (10). We pass it to the kernel as parameter n,
    // because the kernel needs the array length for its bounds check.
    fill<<<blocks, threads>>>(a, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) printf("%d ", a[i]);
    printf("\n");

    cudaFree(a);
}

// Output (threads 10 and 11 exist but write nothing):
// 0 1 2 3 4 5 6 7 8 9

// ================================================================

// 06_flip.cu

// The plan: the CPU makes two arrays and fills "in" with 0 1 2 ... 8.
// The GPU copies every element of "in" to its mirrored position in "out".
// The CPU then prints both arrays and checks the result.
//
// Steps in main():
//   1. Reserve two arrays of 9 ints: in and out.
//   2. Fill in on the CPU with 0 ... 8.
//   3. Choose the launch: 4 threads per block, (9 + 3) / 4 = 3 blocks = 12 threads.
//   4. Launch flip: threads 0..8 each move one element; threads 9..11 are stopped by the bounds check.
//   5. Wait for the GPU.
//   6. Print in and out on the CPU.
//   7. Verify: in[i] was i, so out[i] must be N - 1 - i.
//   8. Free both arrays.
//
// About step 3 — warps:
//   The GPU never runs a single thread on its own. It runs threads in groups of 32, called a warp.
//   All 32 threads of a warp execute the same instruction at the same time.
//   Every block is cut into warps:
//     block of 4 threads   -> 1 warp, of which only 4 lanes work and 28 sit idle
//     block of 256 threads -> 8 full warps, nothing wasted
//   So in real code the block size is a multiple of 32 (typically 256).
//   Here we use 4 only to keep the example and its output small.
//
//   Real code example, for one million elements:
//     #define N 1000000
//     int threads = 256;
//     int blocks = (N + threads - 1) / threads;   -> 3907 blocks
//   3907 * 256 = 1,000,192 threads; the last 192 are stopped by the bounds check.
//   (With N that large you would not print the arrays; the verification loop does the checking.)
//
// Why two arrays: every thread reads from one array and writes to another, so no thread can
// overwrite a value that another thread still needs. The order in which threads run does not matter.

#include <cstdio>

#define N 9

// in  = pointer to the input array (0 1 2 ... 8). const = the kernel promises to only read it.
// out = pointer to the output array; the kernel writes the flipped values here.
// n   = number of elements (9), needed for the mirror position and the bounds check.
__global__ void flip(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // this is the flipping logic: each thread puts its element at the mirrored position.
    if (i < n) out[n - 1 - i] = in[i];
}

int main() {
    int *in, *out;
    cudaMallocManaged(&in, N * sizeof(int));
    cudaMallocManaged(&out, N * sizeof(int));

    // The starting array is made here: the cudaMallocManaged above reserves the memory,
    // and this loop fills it on the CPU with 0 1 2 ... 8.
    for (int i = 0; i < N; i++) in[i] = i;

    int threads = 4;    
    int blocks = (N + threads - 1) / threads; //3
    flip<<<blocks, threads>>>(in, out, N);
    cudaDeviceSynchronize();

    printf("in:  ");
    for (int i = 0; i < N; i++) printf("%d ", in[i]);
    printf("\nout: ");
    for (int i = 0; i < N; i++) printf("%d ", out[i]);
    printf("\n");

    // check
    for (int i = 0; i < N; i++) {
        if (out[i] != N - 1 - i) {
            printf("error at %d\n", i);
            return 1;
        }
    }
    printf("correct\n");

    cudaFree(in);
    cudaFree(out);
}

// Output:
// in:  0 1 2 3 4 5 6 7 8
// out: 8 7 6 5 4 3 2 1 0
// correct

// Figure: First session's exercise: flip an array.  (figures/session_1_figures/slide-23.pdf)

// ================================================================

// 07_flip_inplace.cu

// In-place is an alternative with one advantage and some drawbacks:
//   + Uses ONE array instead of two: half the memory.
//   - Not faster: both versions read N values and write N values in total.
//   - Easier to get wrong: if all N threads simply copy their element to the mirror position,
//     a thread can overwrite a value that another thread has not read yet (a race condition,
//     topic of session 2). That is why only the first half of the threads work, each swapping one pair.
//   - The original input is gone afterwards, so you cannot compare against it.
// It is included mainly to show a race-condition trap.
//
// What is a race condition?
//   Two or more threads access the SAME memory location, at least one of them WRITES,
//   and nothing controls which one goes first. The result then depends on timing: on which
//   thread happens to be faster. It can be right in one run and wrong in the next.
//   On a GPU, thousands of threads run in parallel in no guaranteed order, so this is easy to cause.
//   Analogy: two people withdraw money from the same account at the same moment. Both read
//   balance 100, both subtract 50, both write back 50. One withdrawal is lost.
//   How to avoid it:
//     - make sure every memory location is written by only one thread (as in 06 and the swap below)
//     - or use synchronization / atomic operations (session 2).
//
// The race condition, concretely. The "obvious" in-place kernel would be:
//
//     if (i < n) a[n - 1 - i] = a[i];      // WRONG: all 9 threads, one array
//
// Thread 0 must write a[0] (= 0) into a[8]. Thread 8 must write a[8] (= 8) into a[0].
// Both touch a[0] and a[8], and nobody decides who goes first:
//   - if thread 8 reads a[8] BEFORE thread 0 overwrites it: a[0] = 8   (correct)
//   - if thread 8 reads a[8] AFTER  thread 0 overwrote it:  a[0] = 0   (wrong: the 8 is lost)
// The same happens for every pair. A possible result is 0 1 2 3 4 3 2 1 0 (a mirror, not a flip),
// and it can differ from run to run. That is what a race condition is: the result depends on
// which thread happens to be faster.
//
// The fix below: only n / 2 threads, and each thread does the WHOLE swap of one pair itself.
// Pair (i, n-1-i) is touched by exactly one thread, so no two threads ever touch the same element.
// tmp is a local variable: every thread has its own private copy (in its own register).

#include <cstdio>

#define N 9

__global__ void flip(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 2) {
        int tmp = a[i];
        a[i] = a[n - 1 - i];
        a[n - 1 - i] = tmp;
    }
}

int main() {
    int* a;
    cudaMallocManaged(&a, N * sizeof(int));

    for (int i = 0; i < N; i++) a[i] = i;

    int threads = 4;
    int blocks = (N / 2 + threads - 1) / threads;
    flip<<<blocks, threads>>>(a, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) printf("%d ", a[i]);
    printf("\n");

    cudaFree(a);
}

// Launch: N / 2 = 4 pairs -> (4 + 3) / 4 = 1 block of 4 threads.
// Thread 0 swaps 0<->8, thread 1 swaps 1<->7, thread 2 swaps 2<->6, thread 3 swaps 3<->5.
// The middle element (4) stays where it is.
// Output:
// 8 7 6 5 4 3 2 1 0

// ================================================================

// Expected for next week
//
// A LaTeX-created PDF document with a solution for exercise 08_flip_2d.cu.
//
// 08_flip_2d.cu: flip a 2D array (mirror an image horizontally)
//
// Problem. A 2D array of W columns by H rows (think: an image). Reverse every row:
//   in:   0  1  2  3  4  5          out:  5  4  3  2  1  0
//         6  7  8  9 10 11                11 10  9  8  7  6
//
// Use a 2D grid of 2D blocks: every thread handles one element and gets an x (column) and a y (row).
//
// Goal.
// - Compute x and y for each thread.
// - Convert (x, y) to a position in memory.
// - Bounds check in BOTH directions.
// - Verify the result on the CPU.
