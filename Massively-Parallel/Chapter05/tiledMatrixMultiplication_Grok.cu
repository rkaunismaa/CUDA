#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Macro for CUDA error checking to avoid repetitive code
// Wraps CUDA API calls, prints error details, and exits on failure
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel for tiled matrix multiplication
// Multiplies matrix A (M x K) by matrix B (K x N) to produce C (M x N)
// Uses shared memory tiles to reduce global memory accesses for better performance
__global__ void tiledMatrixMulKernel(float *A, float *B, float *C, int M, int N, int K, int tileSize) {
    // Declare shared memory for tiles of A and B
    // Size is tileSize x tileSize, matching the block dimensions
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    // Get block and thread indices
    int bx = blockIdx.x;  // Block index in x-dimension (columns of C)
    int by = blockIdx.y;  // Block index in y-dimension (rows of C)
    int tx = threadIdx.x; // Thread index within block (x)
    int ty = threadIdx.y; // Thread index within block (y)

    // Calculate global row and column indices for this thread in output matrix C
    int row = by * tileSize + ty;
    int col = bx * tileSize + tx;

    // Initialize partial sum for this thread's element in C
    float sum = 0.0f;

    // Loop over tiles along the K dimension (shared dimension of A and B)
    // Each iteration processes one tile of A and B
    for (int t = 0; t < (K + tileSize - 1) / tileSize; ++t) {
        // Load elements into shared memory tileA
        // Check boundaries to avoid accessing invalid memory
        if (row < M && t * tileSize + tx < K) {
            tileA[ty][tx] = A[row * K + t * tileSize + tx];
        } else {
            tileA[ty][tx] = 0.0f; // Pad with zero if outside matrix
        }

        // Load elements into shared memory tileB
        if (t * tileSize + ty < K && col < N) {
            tileB[ty][tx] = B[(t * tileSize + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f; // Pad with zero if outside matrix
        }

        // Synchronize threads to ensure all shared memory is loaded
        __syncthreads();

        // Compute partial sum for this tile
        // Each thread computes one element of the output tile
        for (int i = 0; i < tileSize; ++i) {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        // Synchronize threads to ensure computation is complete before loading next tile
        __syncthreads();
    }

    // Write result to global memory if within matrix bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to select the most powerful GPU based on compute capability and memory
// Returns the device ID of the selected GPU
int selectMostPowerfulGPU() {
    int deviceCount;
    // Get the number of CUDA-capable devices
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        exit(EXIT_FAILURE);
    }

    int bestDevice = 0;
    int maxComputeCapability = 0;
    size_t maxGlobalMemory = 0;

    // Iterate through all devices to find the best one
    printf("Scanning %d CUDA devices...\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        // Retrieve properties of the current device
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        // Compute capability as a single number (e.g., 7.5 -> 75)
        int computeCapability = prop.major * 10 + prop.minor;
        size_t globalMemory = prop.totalGlobalMem;

        // Print device details for user information
        printf("Device %d: %s, Compute Capability: %d.%d, Global Memory: %zu MB\n",
               i, prop.name, prop.major, prop.minor, globalMemory / (1024 * 1024));

        // Update best device if this one has higher compute capability
        // or same compute capability but more memory
        if (computeCapability > maxComputeCapability ||
            (computeCapability == maxComputeCapability && globalMemory > maxGlobalMemory)) {
            maxComputeCapability = computeCapability;
            maxGlobalMemory = globalMemory;
            bestDevice = i;
        }
    }

    // Inform user of the selected device
    printf("Selecting Device %d with Compute Capability %d.%d and %zu MB memory.\n",
           bestDevice, maxComputeCapability / 10, maxComputeCapability % 10,
           maxGlobalMemory / (1024 * 1024));
    return bestDevice;
}

// Function to determine matrix dimensions and tile size based on available GPU memory
// Sets M, N, K for matrices A (M x K), B (K x N), C (M x N), and tileSize
// Uses a fraction of free memory to avoid over-allocation
void determineMatrixSize(size_t freeMem, int *M, int *N, int *K, int *tileSize) {
    // Use 80% of free memory to account for CUDA overhead and other processes
    freeMem = freeMem * 0.8;
    printf("Adjusted usable memory: %zu MB\n", freeMem / (1024 * 1024));

    // Calculate memory needed for three matrices (A, B, C) of floats
    // Assume square matrices (M = N = K) for simplicity
    size_t memPerMatrix = freeMem / (3 * sizeof(float));
    // Compute maximum matrix dimension (square root of elements per matrix)
    *M = *N = *K = (int)sqrt((double)memPerMatrix);

    // Set tile size to 32 (optimal for most GPUs due to warp size and memory coalescing)
    *tileSize = 32;
    // Adjust matrix dimensions to be multiples of tileSize for efficient tiling
    *M = (*M / *tileSize) * *tileSize;
    *N = *M;
    *K = *M;

    // Verify that the total memory needed does not exceed available memory
    size_t totalMemNeeded = 3 * sizeof(float) * (size_t)(*M) * (size_t)(*N);
    if (totalMemNeeded > freeMem) {
        fprintf(stderr, "Memory calculation error: needed %zu bytes, available %zu bytes\n",
                totalMemNeeded, freeMem);
        exit(EXIT_FAILURE);
    }

    // Print matrix and tile size information
    printf("Matrix dimensions: M=%d, N=%d, K=%d, Tile Size=%d, Memory needed: %zu MB\n",
           *M, *N, *K, *tileSize, totalMemNeeded / (1024 * 1024));
}

int main() {
    // Select the most powerful GPU and set it as the active device
    int device = selectMostPowerfulGPU();
    CUDA_CHECK(cudaSetDevice(device));

    // Query available and total memory on the selected GPU
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("Free memory: %zu MB, Total memory: %zu MB\n",
           freeMem / (1024 * 1024), totalMem / (1024 * 1024));

    // Determine matrix dimensions and tile size based on available memory
    int M, N, K, tileSize;
    determineMatrixSize(freeMem, &M, &N, &K, &tileSize);

    // Calculate sizes for matrices A (M x K), B (K x N), C (M x N)
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory for input and output matrices
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    // Check for host memory allocation failures
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize matrices A and B with random values, C with zeros
    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    // Allocate device memory for matrices A, B, and C with retry mechanism
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    bool allocated = false;
    int attempts = 0;
    const int maxAttempts = 3;

    while (!allocated && attempts < maxAttempts) {
        cudaError_t err;

        // Try allocating matrix A
        err = cudaMalloc(&d_A, sizeA);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate matrix A (%zu MB): %s\n",
                    sizeA / (1024 * 1024), cudaGetErrorString(err));
            goto cleanup;
        }

        // Try allocating matrix B
        err = cudaMalloc(&d_B, sizeB);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate matrix B (%zu MB): %s\n",
                    sizeB / (1024 * 1024), cudaGetErrorString(err));
            goto cleanup;
        }

        // Try allocating matrix C
        err = cudaMalloc(&d_C, sizeC);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate matrix C (%zu MB): %s\n",
                    sizeC / (1024 * 1024), cudaGetErrorString(err));
            goto cleanup;
        }

        allocated = true;
        break;

    cleanup:
        // Free any partially allocated memory
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        d_A = d_B = d_C = NULL;

        // Reduce matrix size by 20% and retry
        M = (int)(M * 0.8 / tileSize) * tileSize;
        N = K = M;
        sizeA = M * K * sizeof(float);
        sizeB = K * N * sizeof(float);
        sizeC = M * N * sizeof(float);
        attempts++;
        printf("Allocation failed. Retrying with smaller size: M=N=K=%d\n", M);
    }

    if (!allocated) {
        fprintf(stderr, "Failed to allocate device memory after %d attempts.\n", maxAttempts);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }

    // Copy input matrices A and B from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    // Block size matches tile size for shared memory efficiency
    dim3 blockDim(tileSize, tileSize);
    // Grid size ensures all elements of C are computed
    dim3 gridDim((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);

    // Print launch configuration for debugging
    printf("Launching kernel with grid (%d, %d), block (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Launch the tiled matrix multiplication kernel
    tiledMatrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, tileSize);
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result matrix C from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify a small subset of results to ensure correctness
    printf("Verifying a sample of results...\n");
    for (int i = 0; i < min(5, M); ++i) {
        for (int j = 0; j < min(5, N); ++j) {
            // Compute expected value on CPU
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            // Check if GPU result matches CPU within floating-point tolerance
            if (fabs(h_C[i * N + j] - sum) > 1e-5) {
                printf("Verification failed at C[%d][%d]: GPU=%f, CPU=%f\n",
                       i, j, h_C[i * N + j], sum);
            }
        }
    }
    printf("Verification completed.\n");

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device to clean up CUDA context
    CUDA_CHECK(cudaDeviceReset());
    printf("Program completed successfully.\n");
    return 0;
}