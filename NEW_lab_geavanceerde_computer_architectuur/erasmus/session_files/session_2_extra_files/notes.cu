// GCA
// session_02
// joris.putteneers@kuleuven.be

// ================================================================


//
// New this session:
// - atomicAdd, atomicSub, atomicMin, atomicMax
// - __syncthreads()
// - __shared__ memory
// - timing with CUDA events
/


//
// Examples:
// 09   the race condition: a sum that gives the wrong answer           (slides 3-4)
// 10   fix 1: atomicAdd                                                (slides 5-6)
// 11   fix 2: reduction, pairing the two halves (every step printed)   (slide 7)
// 11b  fix 2: reduction exactly as in the slide figure (neighbours)    (slide 7)
// 12   reduction for large arrays (shared memory + one atomic per block)
// 13   timing: CPU vs GPU
// 14   exercise + solution: histogram of an image (sessions 1 + 2)
// then the session-2 exercise: max detection using three approaches  (slides 9-10)
//


/*
 * Slide 2-3: Outline
 * - Race conditions
 * - Brute force fixes: atomics, locks, and reduction
 */

// problem statement

/*
 *
 * - Let int *x point to global memory. *x++ happens in 3 steps:
 *     1. Read the value in *x into a register.
 *     2. Add 1 to the value read in step 1.
 *     3. Write the result back to *x.
 * - If we want parallel threads A and B to both increment *x, then we want
 *   something like:
 *     1. Thread A reads the value, 7, from *x.
 *     2. Thread A adds 1 to its value, 7, to make 8.
 *     3. Thread A writes its value, 8, back to *x.
 *     4. Thread B reads the value, 8, from *x.
 *     5. Thread B adds 1 to its value, 8, to make 9.
 *     6. Thread B writes the value, 9, back to *x.
 *
 * Explanation: one increment is not one operation but three. For a correct
 * result, B may only read after A has written.
 *
 * Step 2: the addition happens in the GPU core (SM), not in memory. The
 * value sits in a register and an integer unit (ALU) adds 1 as a normal
 * binary addition. Memory can only be read from and written to, it cannot
 * add. That is why the value has to travel memory -> register -> memory.
 */
//
// Many threads have to combine their results into ONE value (a sum, a maximum, a count).
// If they all update the same memory location, updates get lost: a race condition.
/
/*
 *  Race conditions
 *
 * - But since the threads are parallel, we might actually get:
 *     1. Thread A reads the value, 7, from *x.
 *     2. Thread B reads the value, 7, from *x.
 *     3. Thread A adds 1 to its value, 7, to make 8.
 *     4. Thread A writes its value, 8, back to *x.
 *     5. Thread B adds 1 to its value, 7, to make 8.
 *     6. Thread B writes the value, 8, back to *x.
 *
 * The problem is that B reads after A has read, but before A has written.
 * At that moment A's 8 only exists in A's register; memory still holds 7.
 * So both threads compute 8 and both write 8. B's write overwrites A's.
 * Two increments happened, memory only shows one: one increment is lost.
 */

/*
 * A Example: race_condition.cu
 *
 * Explanation: 1000 blocks x 1000 threads all do *a += 1 on the same int.
 * The kernel was called colonel() on the slides: a pun on "kernel" (same
 * pronunciation). Renamed to increment().
 */

#include <cstdio>

#define N 1024

// total is a pointer to ONE int (not an array): the shared result.
// *total means "the int that total points to".
__global__ void sum(const int* a, int* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) *total += a[i];
}

<-- i think we hsould include some timer functions here so we can output the result (speculation)
int main() {
    int *a, *total;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&total, sizeof(int));

    for (int i = 0; i < N; i++) a[i] = 1;
    *total = 0;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sum<<<blocks, threads>>>(a, total, N);
    cudaDeviceSynchronize();

    printf("expected %d, got %d\n", N, *total);

    cudaFree(a);
    cudaFree(total);
}

// Output:
 *
 * This is the race condition showcased: the result depends on the timing
 * of the threads.
 */

/*
 *
 * - Race condition: A computational hazard that arises when the results of
 *   the program depend on the timing of uncontrollable events, such as the
 *   execution order or threads.
 *     - Many race conditions are caused by violations of the SIMT paradigm.
 *     - Atomic operations and locks are brute force ways to fix race
 *       conditions.
 */


/*

/*
 * Atomics
 *
 * - Atomic operation: an operation that forces otherwise parallel threads
 *   into a bottleneck, executing the operation one at a time.
 *   with an atomic function,
 *   to fix the race condition in race_condition.cu.
 *
/*
 *  race_condition_fixed.cu 
 *
 */

// Fix 1: atomic operation.
// atomicAdd(address, value) does read + add + write as ONE uninterruptible step.
// The hardware effectively locks that memory address while one thread updates it;
// the other threads wait their turn. Threads get SEQUENTIAL access to the shared result.
//
// Result: always correct.
// Cost: 1024 threads all queue for the same address, so the additions happen one after another.
// Correct, but not really parallel anymore. Fine for a few updates, slow for millions.
//
// Other atomic functions work the same way:
//   atomicSub(address, value)
//   atomicMin(address, value)   (int / unsigned int / 64-bit integers only, no float)
//   atomicMax(address, value)   (int / unsigned int / 64-bit integers only, no float)
// They all return the OLD value at the address (we ignore it here).

#include <cstdio>

#define N 1024

__global__ void sum(const int* a, int* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(total, a[i]);
}



<-- i think we hsould include some timer functions here so we can output the result (speculation)
	and compare with the non atomic version
int main() {
    int *a, *total;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&total, sizeof(int));

    for (int i = 0; i < N; i++) a[i] = 1;
    *total = 0;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sum<<<blocks, threads>>>(a, total, N);
    cudaDeviceSynchronize();

    printf("expected %d, got %d\n", N, *total);

    cudaFree(a);
    cudaFree(total);
}

// Output:
// expected 1024, got 1024

*/

// 11_reduction.cu

// Fix 2: reduction. Instead of everyone adding into one place, add PAIRS in parallel, like a tree.
// 16 values, 8 threads, one block:
//   step 1 (stride 8): thread i does a[i] += a[i + 8]   -> 8 partial sums in a[0..7]
//   step 2 (stride 4): thread i does a[i] += a[i + 4]   -> 4 partial sums in a[0..3]
//   step 3 (stride 2): thread i does a[i] += a[i + 2]   -> 2 partial sums in a[0..1]
//   step 4 (stride 1): thread i does a[i] += a[i + 1]   -> the total in a[0]
// Every step halves the number of values: 16 values take log2(16) = 4 steps instead of 16.
// No race: in each step every thread writes its own a[i], and reads an a[i + stride] nobody writes.
//
// __syncthreads(): all threads of the BLOCK wait here until every thread has arrived.
// Needed between steps: step 2 reads values that other threads wrote in step 1.
// Without it, a fast thread could read a[i + 4] before the thread responsible has updated it.
// Rules:
//   - it only synchronizes threads within ONE block (not between blocks)
//   - every thread of the block must reach it, so never put it inside an if that only some threads enter
//
// This simple version needs: N a power of 2, one block, N / 2 threads (so N <= 2048).

#include <cstdio>

#define N 16

__global__ void reduce(int* a) {
    int i = threadIdx.x;
    for (int stride = blockDim.x; stride > 0; stride /= 2) {
        if (i < stride) a[i] += a[i + stride];
        __syncthreads();

        // Only to show the steps: thread 0 prints the partial sums that are still in use.
        // The second __syncthreads() makes the others wait until printing is done.
        if (i == 0) {
            printf("stride %d:", stride);
            for (int k = 0; k < stride; k++) printf(" %d", a[k]);
            printf("\n");
        }
        __syncthreads();
    }
}

int main() {
    int* a;
    cudaMallocManaged(&a, N * sizeof(int));

    for (int i = 0; i < N; i++) a[i] = i;

    reduce<<<1, N / 2>>>(a);
    cudaDeviceSynchronize();

    printf("sum = %d (expected %d)\n", a[0], N * (N - 1) / 2);

    cudaFree(a);
}

// Output (input 0 1 2 ... 15):
// stride 8: 8 10 12 14 16 18 20 22
// stride 4: 24 28 32 36
// stride 2: 56 64
// stride 1: 120
// sum = 120 (expected 120)


// ================================================================

// 11b_reduction_figure.cu  —  the reduction exactly as in the slide figure

// Same idea as 11, but the pairs are formed like in the figure: NEIGHBOURS first.
//   step 1 (stride 1): a[0] += a[1], a[2] += a[3], a[4] += a[5], ...     8 threads
//   step 2 (stride 2): a[0] += a[2], a[4] += a[6], a[8] += a[10], ...    4 threads
//   step 3 (stride 4): a[0] += a[4], a[8] += a[12]                       2 threads
//   step 4 (stride 8): a[0] += a[8]                                      1 thread  -> 41
// Thread t works on position i = 2 * stride * t. Threads whose i falls outside the array do nothing.
// The input is the one from the figure, and every step prints the whole array,
// so the output can be compared row by row with the figure.
//
// 11 versus 11b:
//   11  : pairs the two halves: thread t does a[t] += a[t + stride]      stride 8, 4, 2, 1
//   11b : pairs neighbours:     thread t works on i = 2 * stride * t     stride 1, 2, 4, 8
// Same result, same number of steps (log2(16) = 4): "shifting direction not important".
// In practice 11 is preferred: its active threads work on elements that sit next to each other
// (a[0..stride-1]), while in 11b the elements in use are spread out (2 * stride apart).
// Elements next to each other are faster to access: see the Memory session.

#include <cstdio>

#define N 16

__global__ void reduce(int* a, int n) {
    int t = threadIdx.x;
    for (int stride = 1; stride < n; stride *= 2) {
        int i = 2 * stride * t;
        if (i < n) a[i] += a[i + stride];
        __syncthreads();

        if (t == 0) {
            printf("stride %d:", stride);
            for (int k = 0; k < n; k++) printf(" %3d", a[k]);
            printf("\n");
        }
        __syncthreads();
    }
}

int main() {
    int input[N] = {10, 1, 8, -1, 0, -2, 3, 5, -2, -3, 2, 7, 0, 11, 0, 2};

    int* a;
    cudaMallocManaged(&a, N * sizeof(int));

    int expected = 0;
    printf("input:   ");
    for (int i = 0; i < N; i++) {
        a[i] = input[i];
        expected += input[i];
        printf(" %3d", a[i]);
    }
    printf("\n");

    reduce<<<1, N / 2>>>(a, N);
    cudaDeviceSynchronize();

    printf("sum = %d (expected %d)\n", a[0], expected);

    cudaFree(a);
}

// Output (compare with the figure, row by row):
// input:     10   1   8  -1   0  -2   3   5  -2  -3   2   7   0  11   0   2
// stride 1:  11   1   7  -1  -2  -2   8   5  -5  -3   9   7  11  11   2   2
// stride 2:  18   1   7  -1   6  -2   8   5   4  -3   9   7  13  11   2   2
// stride 4:  24   1   7  -1   6  -2   8   5  17  -3   9   7  13  11   2   2
// stride 8:  41   1   7  -1   6  -2   8   5  17  -3   9   7  13  11   2   2
// sum = 41 (expected 41)

// ================================================================


// 12_reduction_blocks.cu

// Reduction for a LARGE array (here 1,000,000 elements), combining both fixes:
//   - every block reduces its own chunk of the array (fast, in shared memory)
//   - thread 0 of each block adds the block's result to the total with ONE atomicAdd
// So 1954 atomic operations instead of 1,000,000.
//
// Why not one block like in 11? A block has at most 1024 threads, each handling 2 elements,
// so one block can reduce at most 2048 elements. And __syncthreads() cannot synchronize
// different blocks, so the blocks must be combined in another way: here with atomicAdd.
//
// __shared__ memory:
//   - small, very fast memory on the GPU chip itself, one piece per block
//   - all threads of the block can read and write it; other blocks cannot see it
//   - it exists only while the block runs
//   Reduction reads and writes the same values many times, so doing that in shared memory is much
//   faster than in the large (global) GPU memory.
//
// Each block handles 2 * 256 = 512 elements. Elements beyond the end of the array are filled
// with 0, which does not change a sum. (For a maximum you would fill with the smallest int, INT_MIN.)

#include <cstdio>

#define N 1000000
#define THREADS 256

__global__ void reduce(const int* a, int* total, int n) {
    __shared__ int s[2 * THREADS];
    int t = threadIdx.x;
    int start = blockIdx.x * 2 * blockDim.x;

    s[t] = (start + t < n) ? a[start + t] : 0;
    s[t + blockDim.x] = (start + t + blockDim.x < n) ? a[start + t + blockDim.x] : 0;
    __syncthreads();

    for (int stride = blockDim.x; stride > 0; stride /= 2) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }

    if (t == 0) atomicAdd(total, s[0]);
}

int main() {
    int *a, *total;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&total, sizeof(int));

    for (int i = 0; i < N; i++) a[i] = 1;
    *total = 0;

    int blocks = (N + 2 * THREADS - 1) / (2 * THREADS);
    reduce<<<blocks, THREADS>>>(a, total, N);
    cudaDeviceSynchronize();

    printf("blocks %d, expected %d, got %d\n", blocks, N, *total);

    cudaFree(a);
    cudaFree(total);
}

// (cond) ? x : y   means: if cond is true use x, otherwise y. A one-line if/else.
//
// Output:
// blocks 1954, expected 1000000, got 1000000

// ================================================================

// 13_timing.cu

// The exercise asks to compare execution times. Two clocks:
//   - CPU code: std::chrono (a normal C++ timer)
//   - GPU code: CUDA events. A kernel launch is asynchronous, so a CPU timer around the launch
//     only measures how long it takes to START the kernel. Events are timestamps recorded
//     by the GPU itself, in order with the kernels.
//
// Order for GPU timing:
//   cudaEventRecord(start) -> launch kernel -> cudaEventRecord(stop)
//   -> cudaEventSynchronize(stop) (wait until the GPU reached "stop") -> cudaEventElapsedTime
//
// Warm-up: the first launch also includes one-time costs (initialization, moving the managed
// memory to the GPU). Launch once before timing so you measure the kernel itself.

#include <cstdio>
#include <chrono>

#define N 1000000

__global__ void sum(const int* a, int* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(total, a[i]);
}

int main() {
    int *a, *total;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&total, sizeof(int));

    for (int i = 0; i < N; i++) a[i] = 1;

    auto c0 = std::chrono::steady_clock::now();
    int cpu = 0;
    for (int i = 0; i < N; i++) cpu += a[i];
    auto c1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    sum<<<blocks, threads>>>(a, total, N);
    cudaDeviceSynchronize();
    *total = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sum<<<blocks, threads>>>(a, total, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    printf("CPU: sum %d in %.3f ms\n", cpu, cpu_ms);
    printf("GPU: sum %d in %.3f ms\n", *total, gpu_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a);
    cudaFree(total);
}

// Output (the times depend on the machine):
// CPU: sum 1000000 in ... ms
// GPU: sum 1000000 in ... ms



/*
 * CUDA C++ atomic functions
 *
 * - Recommended: cuda::atomic_ref (libcu++, #include <cuda/atomic>)
 *     fetch_add                         (legacy: atomicAdd)
 *     fetch_sub                         (legacy: atomicSub)
 *     fetch_min / fetch_max             (legacy: atomicMin / atomicMax)
 *     fetch_and / fetch_or / fetch_xor  (legacy: atomicAnd / atomicOr / atomicXor)
 *     exchange                          (legacy: atomicExch)
 *     compare_exchange_strong           (legacy: atomicCAS)
 * - For documentation, refer to the CUDA Programming Guide.
 *

/*
 *  atomicCAS(int *address, int compare, int val): needed for locks
 *
 *   1. Read the value, old, located at address.
 *   2. *address = (old == compare) ? val : old;
 *   3. Return old.
 *
 * Explanation: CAS = Compare-And-Swap. "If *address is still compare,
 * replace it with val." The 3 steps happen as one indivisible step.
 * The return value tells you if it worked: old == compare means you
 * replaced the value.
 *

/*
 * Slide 16: Locks and mutex
 *
 * - Lock: a mechanism in parallel computing that forces an entire segment
 *   of code to be executed atomically.
 * - mutex
 *     - "mutual exclusion", the principle behind locks.
 *     - While a thread is running code inside a lock, it shuts all the other
 *       threads out of the lock.
 *
 *   __global__ void someKernel(void){
 *       Lock myLock;
 *
 *       // some parallel code
 *
 *       myLock.lock();
 *       // some sequential code
 *       myLock.unlock();
 *
 *       // some parallel code
 *   }
 *
 * An atomic makes one operation indivisible; a lock makes a whole piece of
 * code indivisible. Only one thread at a time can be between lock() and
 * unlock().
 *
 */

/*
 * Slide 17: The concept
 *
 * [image: session_2_img5.png]
 *
 *   Thread A: gets to the lock first and reserves it
 *             -> does instruction set I (atomic)
 *             -> unlocks the lock
 *             -> does instruction set II (parallel)
 *
 *   Thread B: gets to the lock second and gets stuck in the lock's loop
 *             -> waits for thread A to unlock the lock
 *             -> reserves the lock
 *             -> does instruction set I (atomic)
 */

/*
 * Slide 18: Lock.h (simplified)
 */


we can have a better example with locks i think? somethjing thats better aligned with the prev examples
can also just be in layout 


#include <cstdio>
#define NBLOCKS 512
#define NTHREADS 1024

__global__ void blockCounterUnlocked(int* nblocks){
	if(threadIdx.x == 0){
		*nblocks = *nblocks + 1;
	}
}

__global__ void blockCounter1(Lock lock, int* nblocks){
	if(threadIdx.x == 0){
		lock.lock();
		*nblocks = *nblocks + 1;
		lock.unlock();
	}
}

int main(){
	int* nblocks;
	Lock lock;
	cudaMallocManaged(&nblocks, sizeof(int));

	*nblocks = 0;
	blockCounterUnlocked<<<NBLOCKS, NTHREADS>>>(nblocks);
	cudaDeviceSynchronize();
	printf("blockCounterUnlocked counted %d blocks\n", *nblocks);

	*nblocks = 0;
	blockCounter1<<<NBLOCKS, NTHREADS>>>(lock, nblocks);
	cudaDeviceSynchronize();
	printf("blockCounter1 counted %d blocks\n", *nblocks);

	cudaFree(nblocks);
	cudaFree(lock.mutex);
}



/*
 * Slide 24: output
 *
 * blockCounterUnlocked counted 47 blocks
 * blockCounter1 counted 512 blocks
 *
 * Explanation:
 * - In both kernels only thread 0 of each block adds 1, so the correct
 *   answer is 512 (the number of blocks).
 * - blockCounterUnlocked: thread 0 of all 512 blocks does read / add / write
 *   on the same int at the same time: race condition, 47 instead of 512.
 * - blockCounter1: the lock lets only one block's thread 0 do the increment
 *   at a time: 512, correct.
 * - Time in the slides: 0.057920 ms unlocked, 0.636064 ms locked, about 11x
 *   slower.
 * - For one increment, atomicAdd(nblocks, 1) is enough. The lock is shown
 *   because it works for any amount of code between lock() and unlock().
 *
 * Differences with the slide version: same as before (cudaMallocManaged,
 * no timers, no cudaMemcpy). NBLOCKS_TRUE / NTHREADS_TRUE renamed to
 * NBLOCKS / NTHREADS; NTHREADS_TRUE was 512 * 2 = 1024.
 */

/*
 * Slide 25: blockCounter.cu pauses indefinitely with this kernel when the
 * threads of a warp run in lockstep
 */


/*
 *
 *
 *
 * does this warp section still belong here? i feel it deserved a better transition
 * - Why? warps.
 *
 * Explanation:
 * - Now every thread calls lock(), not only thread 0.
 * - A warp is 32 threads that execute the same instruction together.
 * - One thread of the warp wins atomicCAS and should leave the loop. The
 *   other 31 lose and stay in the loop. In lockstep the warp cannot run
 *   both paths at once: it keeps running the loop for the 31, and the
 *   winner waits for them, so it never reaches unlock(). The 31 wait for
 *   the winner to unlock. Nobody can continue: the program hangs
 *   (deadlock).
 * - Current GPUs can schedule the threads of a warp independently, so the
 *   winner can reach unlock() and the kernel may finish (not tested).
 */


/*
// exercise — histogram of an image (sessions 1 + 2)
//
// A grayscale "image" of W x H pixels (for example 640 x 480). Every pixel has a value from 0 to 9.
// Count how many pixels have each value: a histogram with 10 bins.
// Fill the image with any pattern you like, for example value = (3x + 5y) % 10.
//
// 1. CPU version: loop over all pixels and do hist[value]++. This is your reference.
//
// 2. GPU version, using what you learned in session 1:
//    - a 2D grid of 2D blocks (for example 16 x 16 threads), one thread per pixel
//    - compute x and y, and the position in memory y * W + x
//    - bounds check in both directions
//    - every thread does hist[value]++
//
// 3. Compare with the CPU:
//    - Is every bin correct?
//    - Is the sum of all bins equal to W * H?
//    - Run it several times. Do you get the same result every time?
//
// 4. Explain what goes wrong. Which threads access the same memory location, and when?
//
// 5. Fix it with atomicAdd (session 2) and verify again.
//
// 6. Bonus:
//    - Time the CPU, the broken and the fixed version.
//    - What happens to the time if all pixels have the same value?
//    - And if you use 256 bins instead of 10?
//
// Questions:
// - Why did the flip from session 1 have no race condition, while the histogram does?
// - Would the race also happen if you launched only one thread?

// ================================================================

// 14_histogram.cu  —  SOLUTION (teacher)

// One program, two kernels, so the race condition is visible side by side:
//   hist_race: every thread does hist[value]++          -> race condition, counts are too low
//   hist_ok:   every thread does atomicAdd(&hist[value], 1) -> correct
//
// The image: W x H pixels, value = (3x + 5y) % 10. Every row contains each value exactly 64 times,
// so every bin must end up at 64 * 480 = 30720, and all bins together at W * H = 307200.

#include <cstdio>

#define W 640
#define H 480
#define BINS 10

__global__ void hist_race(const int* img, int* hist, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) hist[img[y * w + x]]++;
}

__global__ void hist_ok(const int* img, int* hist, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) atomicAdd(&hist[img[y * w + x]], 1);
}

int main() {
    int *img, *race, *ok;
    cudaMallocManaged(&img, W * H * sizeof(int));
    cudaMallocManaged(&race, BINS * sizeof(int));
    cudaMallocManaged(&ok, BINS * sizeof(int));
    int cpu[BINS] = {0};

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y * W + x] = (3 * x + 5 * y) % BINS;

    for (int i = 0; i < W * H; i++) cpu[img[i]]++;

    for (int b = 0; b < BINS; b++) {
        race[b] = 0;
        ok[b] = 0;
    }

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);
    hist_race<<<blocks, threads>>>(img, race, W, H);
    hist_ok<<<blocks, threads>>>(img, ok, W, H);
    cudaDeviceSynchronize();

    int sum_race = 0, sum_ok = 0;
    printf("bin      cpu     race       ok\n");
    for (int b = 0; b < BINS; b++) {
        printf("%3d %8d %8d %8d\n", b, cpu[b], race[b], ok[b]);
        sum_race += race[b];
        sum_ok += ok[b];
    }
    printf("sum %8d %8d %8d\n", W * H, sum_race, sum_ok);

    cudaFree(img);
    cudaFree(race);
    cudaFree(ok);
}

// Output (the "race" column is lower and changes from run to run):
// bin      cpu     race       ok
//   0    30720      ...    30720
//   1    30720      ...    30720
//   ...
//   9    30720      ...    30720
// sum   307200      ...   307200
//
// Why it races: neighbouring pixels often have the same value, so many threads (even within one
// warp) do hist[v]++ on the SAME bin at the same time. Each ++ is read + add + write; threads read
// the same old count, both add 1, both write back -> one increment is lost.
//
// Why the flip from session 1 had no race: there every thread wrote to its OWN output position,
// computed from its thread index. Here the write position depends on the DATA (the pixel value),
// so different threads can end up writing to the same place.

/

// reading list
// unified memory: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#unified-memory
// warps: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
// atomics: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomicadd
// cuda::atomic_ref: https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html
// series materials: https://wlandau.github.io/gpu/
