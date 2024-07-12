// Friday, July 12, 2024
// https://claude.ai/chat/f21125f2-40a4-4bf1-a799-6be5cc4694c7
// 1) "Create a CUDA program that demonstrates the Optimized Parallel Tiled Approach to perform 2D Convolution by taking advantage of the lower latency, higher bandwidth shared memory as well as global constant memory cached aggresively within GPU thread blocks."
// 2) "Please explain and break down the code."
// 3) "Add judicious comments to the code explaining key concepts."
// 4) "Create a make file to compile this program for CUDA."
// 5) "Modify this make file to include debugging information."

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Define tile size for optimal performance (usually a multiple of 32 for modern GPUs)
#define TILE_SIZE 16
#define MASK_SIZE 3
#define MASK_RADIUS MASK_SIZE / 2

// Declare convolution mask in constant memory for faster access and broadcasting
__constant__ float d_Mask[MASK_SIZE * MASK_SIZE];

// CUDA kernel for 2D convolution using tiled approach
__global__ void convolution2DTiled(float* d_Input, float* d_Output, int width, int height) {
    // Declare shared memory for input tile, including halo region
    __shared__ float s_Input[TILE_SIZE + MASK_SIZE - 1][TILE_SIZE + MASK_SIZE - 1];

    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int x = bx + tx;
    int y = by + ty;

    // Load input tile into shared memory, including halo region
    for (int i = ty; i < TILE_SIZE + MASK_SIZE - 1; i += TILE_SIZE) {
        for (int j = tx; j < TILE_SIZE + MASK_SIZE - 1; j += TILE_SIZE) {
            int x_in = bx + j - MASK_RADIUS;
            int y_in = by + i - MASK_RADIUS;
            
            // Handle boundary conditions
            if (x_in >= 0 && x_in < width && y_in >= 0 && y_in < height) {
                s_Input[i][j] = d_Input[y_in * width + x_in];
            } else {
                s_Input[i][j] = 0.0f;
            }
        }
    }

    // Ensure all threads have loaded their data before proceeding
    __syncthreads();

    // Perform convolution for threads within image boundaries
    if (x < width && y < height) {
        float sum = 0.0f;
        // Iterate over the mask
        for (int i = 0; i < MASK_SIZE; i++) {
            for (int j = 0; j < MASK_SIZE; j++) {
                // Multiply input pixel with corresponding mask value and accumulate
                sum += s_Input[ty + i][tx + j] * d_Mask[i * MASK_SIZE + j];
            }
        }
        // Write the result to global memory
        d_Output[y * width + x] = sum;
    }
}

// Host function to initialize the convolution mask
void initializeMask(float* mask) {
    // Simple averaging mask for demonstration
    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
        mask[i] = 1.0f / (MASK_SIZE * MASK_SIZE);
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int size = width * height * sizeof(float);

    // Allocate host memory
    float* h_Input = (float*)malloc(size);
    float* h_Output = (float*)malloc(size);
    float* h_Mask = (float*)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));

    // Initialize input with random values and set up the mask
    for (int i = 0; i < width * height; i++) {
        h_Input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    initializeMask(h_Mask);

    // Allocate device memory
    float* d_Input, * d_Output;
    cudaMalloc(&d_Input, size);
    cudaMalloc(&d_Output, size);

    // Copy input data to device
    cudaMemcpy(d_Input, h_Input, size, cudaMemcpyHostToDevice);
    // Copy mask to constant memory
    cudaMemcpyToSymbol(d_Mask, h_Mask, MASK_SIZE * MASK_SIZE * sizeof(float));

    // Set up grid and block dimensions for kernel launch
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the convolution kernel
    convolution2DTiled<<<dimGrid, dimBlock>>>(d_Input, d_Output, width, height);

    // Copy result back to host
    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyDeviceToHost);

    // Clean up: Free device and host memory
    cudaFree(d_Input);
    cudaFree(d_Output);
    free(h_Input);
    free(h_Output);
    free(h_Mask);

    return 0;
}