// Monday, June 24, 2024
// Generated by ChatGPT
// https://chatgpt.com/c/beb320a9-c8cb-49eb-9996-3820bf1a1b45

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for tiled matrix multiplication with non-square matrices
__global__ void matrixMulTiled(float *d_A, float *d_B, float *d_C, int A_height, int A_width, int B_width) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    for (int m = 0; m < (A_width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < A_height && m * TILE_WIDTH + tx < A_width)
            ds_A[ty][tx] = d_A[Row * A_width + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0;

        if (m * TILE_WIDTH + ty < A_width && Col < B_width)
            ds_B[ty][tx] = d_B[(m * TILE_WIDTH + ty) * B_width + Col];
        else
            ds_B[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }
        
        __syncthreads();
    }

    if (Row < A_height && Col < B_width)
        d_C[Row * B_width + Col] = Cvalue;
}

void randomInit(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

int main() {
    
    int A_height = 8192;
    int A_width = 4096;
    int B_width = 8192;
    
    size_t size_A = A_height * A_width * sizeof(float);
    size_t size_B = A_width * B_width * sizeof(float);
    size_t size_C = A_height * B_width * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Initialize matrices h_A and h_B with random values
    srand(42); // Seed for reproducibility
    randomInit(h_A, A_height * A_width);
    randomInit(h_B, A_width * B_width);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((B_width + TILE_WIDTH - 1) / TILE_WIDTH, (A_height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, A_height, A_width, B_width);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize to wait for the stop event to complete
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Convert to minutes:seconds
    int minutes = static_cast<int>(milliseconds / 1000) / 60;
    int seconds = static_cast<int>(milliseconds / 1000) % 60;

    // Output the timing result
    printf("Time elapsed: %d minutes %d seconds\n", minutes, seconds);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Optionally check the result (for example, print some of the values)
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}