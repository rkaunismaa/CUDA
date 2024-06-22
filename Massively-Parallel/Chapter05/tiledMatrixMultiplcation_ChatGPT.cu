// Saturday, June 22, 2024
// Another example generated b ChatGPT!
// https://chatgpt.com/c/beb320a9-c8cb-49eb-9996-3820bf1a1b45

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for tiled matrix multiplication
__global__ void matrixMulTiled(float *d_A, float *d_B, float *d_C, int width) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    for (int m = 0; m < (width / TILE_WIDTH); ++m) {
        ds_A[ty][tx] = d_A[Row * width + m * TILE_WIDTH + tx];
        ds_B[ty][tx] = d_B[(m * TILE_WIDTH + ty) * width + Col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }
        
        __syncthreads();
    }
    
    d_C[Row * width + Col] = Cvalue;
}

int main() {

    // int width = 18432; // Adjusted size for 4GB VRAM usage
    int width = 36864; // Adjusted size for 8GB VRAM usage 2 x 18432
    //int width = 73728; // Adjusted size for 16GB VRAM usage 4 x 18432

    size_t size = width * width * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices h_A and h_B
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f; // Example initialization
        h_B[i] = 1.0f; // Example initialization
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);

    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Check the result (for example, print some of the values)
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
