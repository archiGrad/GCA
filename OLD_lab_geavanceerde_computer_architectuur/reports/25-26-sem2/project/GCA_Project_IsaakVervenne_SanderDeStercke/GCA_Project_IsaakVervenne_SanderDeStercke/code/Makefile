# ==============================================================================
# COMPILERS
# ==============================================================================
# NVCC is Nvidia's CUDA compiler. It handles the .cu files.
NVCC = nvcc
# CXX is the standard C++ compiler (GNU). It handles the .cpp files.
CXX  = g++

# ==============================================================================
# GPU (CUDA) FLAGS
# ==============================================================================
# -O3        : Maximum compiler optimization for speed (crucial for performance testing).
# -std=c++17 : Uses the C++17 standard.
# -lineinfo  : Generates line-number information for profiling tools (like Nsight Compute).
# -Xptxas=-v : Tells the PTX assembler to print memory usage (registers, shared memory) per kernel. 
# 
# -arch=sm_XX: THIS IS THE MOST IMPORTANT PARAMETER TO TWEAK. 
# It tells the compiler which GPU architecture to target. Choose the one matching your GPU:
#   sm_75 -> Turing (e.g., RTX 20-series, GTX 16-series)
#   sm_86 -> Ampere (e.g., RTX 30-series, A-series workstations)
#   sm_89 -> Ada Lovelace (e.g., RTX 40-series)
#   sm_90 -> Hopper (e.g., H100)
NVCC_FLAGS = -O3 -lineinfo -Xptxas=-v -arch=sm_75 -std=c++17

# ==============================================================================
# CPU FLAGS
# ==============================================================================
# CPU Single-core: Just maximum optimization and C++17.
CXX_FLAGS_SINGLE = -O3 -std=c++17

# CPU Multi-core: 
# -fopenmp enables OpenMP parallelization (tells the compiler to process the #pragma omp directives).
CXX_FLAGS_MULTI  = -O3 -std=c++17 -fopenmp


# ==============================================================================
# BUILD TARGETS
# ==============================================================================
# Typing 'make' or 'make all' will compile all 4 versions of your simulation.
all: fpv_cfd fpv_cfd_stl fpv_cfd_cpu_single fpv_cfd_cpu_multi

# 1. GPU version with hardcoded arm geometry
fpv_cfd: fpv_cfd.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# 2. GPU version reading the external STL-derived mask
fpv_cfd_stl: fpv_cfd_stl.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# 3. CPU single-core version (Baseline reference)
fpv_cfd_cpu_single: fpv_cfd_cpu_single.cpp
	$(CXX) $(CXX_FLAGS_SINGLE) -o $@ $<

# 4. CPU multi-core version using OpenMP
fpv_cfd_cpu_multi: fpv_cfd_cpu_multi.cpp
	$(CXX) $(CXX_FLAGS_MULTI) -o $@ $<

# ==============================================================================
# CLEAN UP
# ==============================================================================
# Typing 'make clean' removes all compiled executables and generated CSV files.
clean:
	rm -f fpv_cfd fpv_cfd_cpu fpv_cfd_stl fpv_cfd_cpu_single fpv_cfd_cpu_multi \
	      velocity.csv velocity_stl*.csv velocity_cpu_single.csv velocity_cpu_multi.csv \
	      obstacle_mask_frame*.csv
	rm -rf images