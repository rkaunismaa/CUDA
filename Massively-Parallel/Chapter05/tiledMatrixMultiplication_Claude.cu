// Tuesday, July 9, 2024
// https://claude.ai/chat/6a75fd89-2603-4b25-b6ae-2d22eef251ed

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Macro for CUDA error checking
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t result = call; \
        if (result != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, result, cudaGetErrorString(result), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for tiled matrix multiplication
__global__ void tiledMatrixMul(float *A, float *B, float *C, int M, int N, int K, int TILE_SIZE)
{
    // Shared memory for the tiles of A and B
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];

    // Calculate global and local indices
    int bx = blockIdx.x;    // block index along x
    int by = blockIdx.y;    // block index along y
    int tx = threadIdx.x;   // thread index along x within a block
    int ty = threadIdx.y;   // thread index along y within a block

    // Calculate global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Loop over all tiles
    for (int t = 0; t < (N - 1) / TILE_SIZE + 1; ++t) {
        // Load elements of A into shared memory
        if (row < M && t * TILE_SIZE + tx < N)
            sA[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;

        // Load elements of B into shared memory
        if (t * TILE_SIZE + ty < N && col < K)
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * K + col];
        else
            sB[ty][tx] = 0.0f;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute dot product between row of sA and column of sB
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sA[ty][i] * sB[i][tx];

        // Synchronize to make sure that the preceding computation is done
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < K)
        C[row * K + col] = sum;
}

// Function to initialize a matrix with random float values
void initializeRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;  // Random float between 0 and 1
    }
}

int main(int argc, char* argv[])
{
    // Check if the correct number of command-line arguments is provided
    if (argc != 5) {
        printf("Usage: %s <TILE_SIZE> <M> <N> <K>\n", argv[0]);
        return 1;
    }

    // Parse command-line arguments
    int TILE_SIZE = atoi(argv[1]);  // Size of each tile
    int M = atoi(argv[2]);  // Number of rows in A
    int N = atoi(argv[3]);  // Number of columns in A / rows in B
    int K = atoi(argv[4]);  // Number of columns in B

    // Calculate sizes of matrices in bytes
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Seed the random number generator
    srand(time(NULL));

    // Initialize host arrays with random values
    initializeRandomMatrix(h_A, M * N);
    initializeRandomMatrix(h_B, N * K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));

    // Copy host memory to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((K - 1) / TILE_SIZE + 1, (M - 1) / TILE_SIZE + 1);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, NULL));

    // Launch kernel
    tiledMatrixMul<<<grid, threads>>>(d_A, d_B, d_C, M, N, K, TILE_SIZE);

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, NULL));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate and print the elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    //printf("Kernel execution time: %f milliseconds\n", milliseconds);

    // Convert to minutes:seconds
    int minutes = static_cast<int>(milliseconds / 1000) / 60;
    int seconds = static_cast<int>(milliseconds / 1000) % 60;

    // Output the timing result
    printf("Tile Width: %d, Time elapsed: %d minutes %d seconds\n", TILE_SIZE, minutes, seconds);

    // Destroy the events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Verify result (simple check: ensure all values are non-negative)
    bool correct = true;
    for (int i = 0; i < M * K; ++i) {
        if (h_C[i] < 0) {
            correct = false;
            break;
        }
    }
    printf("Matrix multiplication completed. All values non-negative: %s\n", correct ? "Yes" : "No");

    // Print a small subset of the result for verification
    printf("Subset of the result matrix:\n");
    for (int i = 0; i < min(5, M); ++i) {
        for (int j = 0; j < min(5, K); ++j) {
            printf("%f ", h_C[i * K + j]);
        }
        printf("\n");
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

// ./tiledMatrixMultiplication_Claude 32, 36864, 36864, 36864
// Tile Width: 32, Time elapsed: 3 minutes 58 seconds
// Matrix multiplication completed. All values non-negative: Yes
// Subset of the result matrix:
// 9207.848633 9186.248047 9209.159180 9173.769531 9221.466797 
// 9238.849609 9251.534180 9257.807617 9167.609375 9219.050781 
// 9161.959961 9184.829102 9173.823242 9127.732422 9186.801758 
// 9262.590820 9252.227539 9293.467773 9207.738281 9271.421875 
// 9205.641602 9175.474609 9242.205078 9124.767578 9242.376953 