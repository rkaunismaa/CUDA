// Tuesday, July 9, 2024
// https://claude.ai/chat/6a75fd89-2603-4b25-b6ae-2d22eef251ed

// Saturday, September 7, 2024
// Running this on the kitchen computer that has the 2070 super installed as the main and only gpu ... 
//     "program": "${workspaceFolder}/Massively-Parallel/Chapter05/tiledMatrixMultiplication_Claude_2",
 //    "args" : "32, 16384, 16384, 16384, 1" 
// Running tiled matrix multiplication kernel
// Kernel execution time : 0 minutes 0 seconds
// Matrix multiplication completed. All values non-negative: Yes
// Subset of the result matrix:
// 5
// .000000 0.000000 0.000000 0.000000 0.000000 



// Tuesday, August 27, 2024
// "program": "${workspaceFolder}/Massively-Parallel/Chapter05/tiledMatrixMultiplication_Claude_2",
// "args" : "32, 36864, 36864, 36864, 1"
// Running tiled matrix multiplication kernel
// Kernel execution time : 3 minutes 58 seconds
// Matrix multiplication completed. All values non-negative: Yes
// Subset of the result matrix:
// .360352 9108.003906 9184.590820 9139.760742 9181.508789 
// .875977 9169.104492 9235.666992 9225.882812 9243.696289 
// .279297 9102.749023 9211.815430 9148.763672 9148.818359 
// .959961 9187.512695 9273.405273 9203.490234 9266.882812 
// .605469 9148.897461 9232.327148 9157.536133 9205.351562 


// ./tiledMatrixMultiplication_Claude_2 32, 36864, 36864, 36864, 1
// Running tiled matrix multiplication kernel
// Kernel execution time : 3 minutes 59 seconds
// Matrix multiplication completed. All values non-negative: Yes
// Subset of the result matrix:
// 9256.574219 9212.584961 9292.360352 9247.765625 9243.450195 
// 9189.045898 9175.709961 9244.813477 9210.253906 9244.516602 
// 9259.499023 9231.224609 9306.111328 9245.775391 9283.497070 
// 9263.297852 9171.152344 9255.652344 9187.128906 9224.153320 
// 9271.532227 9203.696289 9292.383789 9222.396484 9274.501953 

// // ./tiledMatrixMultiplication_Claude_2 32, 36864, 36864, 36864, 0
// Running non-tiled matrix multiplication kernel
// Kernel execution time : 3 minutes 55 seconds
// Matrix multiplication completed. All values non-negative: Yes
// Subset of the result matrix:
// 9180.284180 9146.212891 9172.700195 9181.636719 9172.363281 
// 9237.592773 9208.097656 9214.133789 9234.019531 9233.512695 
// 9236.725586 9175.346680 9172.720703 9211.685547 9218.524414 
// 9350.291016 9300.338867 9290.190430 9310.751953 9295.728516 
// 9203.264648 9136.223633 9159.518555 9163.311523 9195.881836 


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

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

// Tiled matrix multiplication kernel
__global__ void tiledMatrixMul(float *A, float *B, float *C, int M, int N, int K, int TILE_SIZE)
{
    // Shared memory for the tiles of A and B
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];

    // Calculate global and local indices
    int bx = blockIdx.x;  // block index along x
    int by = blockIdx.y;  // block index along y
    int tx = threadIdx.x; // thread index along x within a block
    int ty = threadIdx.y; // thread index along y within a block

    // Calculate global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Iterate over all tiles needed to compute the matrix multiplication
    for (int t = 0; t < (N - 1) / TILE_SIZE + 1; ++t) {
        // Load elements from matrix A into shared memory
        if (row < M && t * TILE_SIZE + tx < N)
            sA[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;

        // Load elements from matrix B into shared memory
        if (t * TILE_SIZE + ty < N && col < K)
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * K + col];
        else
            sB[ty][tx] = 0.0f;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Perform dot product between row of sA and column of sB
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sA[ty][i] * sB[i][tx];

        // Synchronize to make sure that the preceding computation is done
        __syncthreads();
    }

    // Write the final result to global memory
    if (row < M && col < K)
        C[row * K + col] = sum;
}

// Non-tiled matrix multiplication kernel
__global__ void simpleMatrixMul(float *A, float *B, float *C, int M, int N, int K)
{
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        // Perform dot product between row of A and column of B
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
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
    if (argc != 6) {
        printf("Usage: %s <TILE_SIZE> <M> <N> <K> <kernel_type>\n", argv[0]);
        printf("kernel_type: 0 for non-tiled, 1 for tiled\n");
        return 1;
    }

    // Parse command-line arguments
    int TILE_SIZE = atoi(argv[1]);  // Size of each tile
    int M = atoi(argv[2]);  // Number of rows in A
    int N = atoi(argv[3]);  // Number of columns in A / rows in B
    int K = atoi(argv[4]);  // Number of columns in B
    int kernel_type = atoi(argv[5]);  // 0 for non-tiled, 1 for tiled

    // Calculate sizes of matrices in bytes
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Seed the random number generator and initialize matrices
    srand(time(NULL));
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

    // Set up execution parameters based on kernel type
    dim3 threads, grid;
    if (kernel_type == 1) {
        // For tiled kernel
        threads = dim3(TILE_SIZE, TILE_SIZE);
        grid = dim3((K - 1) / TILE_SIZE + 1, (M - 1) / TILE_SIZE + 1);
    } else {
        // For non-tiled kernel
        threads = dim3(32, 32);
        grid = dim3((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, NULL));

    // Launch the appropriate kernel
    if (kernel_type == 1) {
        printf("Running tiled matrix multiplication kernel\n");
        tiledMatrixMul<<<grid, threads>>>(d_A, d_B, d_C, M, N, K, TILE_SIZE);
    } else {
        printf("Running non-tiled matrix multiplication kernel\n");
        simpleMatrixMul<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
    }

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
    printf("Kernel execution time : %d minutes %d seconds\n", minutes, seconds);

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