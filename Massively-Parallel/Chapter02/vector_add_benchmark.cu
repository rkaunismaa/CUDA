// Thursday, July 4, 3034
// https://chatgpt.com/c/03215fe4-105b-4e0a-97a4-26e3a528611f
// I asked ChatGPT these questions, and below is the final code sample.
// 1) "Please create a cuda program that demonstrates vector addition. "
// 2) "Initialise h_A and h_B with random floats."
// 3) "Why did you specify a blockSize of 256?"
// 4) "What are the optimal values for a 4090 gpu?"
// "This program will run the vector addition kernel with different block sizes and print the execution time for each. 
//  You can use this information to determine the optimal block size for your specific GPU and application."

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Function to initialize vectors with random floats
void initializeVectors(float *A, float *B, int N) {
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to perform vector addition and measure execution time
void benchmarkVectorAdd(int N, int blockSize) {
    size_t size = N * sizeof(float);
    
    // Allocate memory on host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize vectors on host
    initializeVectors(h_A, h_B, N);

    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid size
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Calculate and print execution time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Block size: " << blockSize << ", Time: " << elapsed.count() << " seconds" << std::endl;
}

int main(int argc, char *argv[]) {

    // if (argc != 2) {
    //     std::cerr << "Usage: " << argv[0] << " <exponent>" << std::endl;
    //     return 1;
    // }
    int P = (argc> 1) ? atoi(argv[1]) : 24; 

    if (P <= 0) {
        std::cerr << "Exponent must be a positive integer." << std::endl;
        return 1;
    }

    int N = 1 << P; // Size of vectors

    // Test different block sizes
    int blockSizes[] = {128, 256, 512, 1024};
    for (int blockSize : blockSizes) {
        benchmarkVectorAdd(N, blockSize);
    }

    return 0;
}
