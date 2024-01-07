#include <stdio.h>

#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


//const int DSIZE = 32*1048576;
//const int DSIZE = 256*1048576;
const int DSIZE = 512*1048576;
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < ds; idx+=gridDim.x*blockDim.x)         // a grid-stride loop
    C[idx] = A[idx] + B[idx]; // do the vector (element) add here
}

int main(){

  auto start_time = std::chrono::high_resolution_clock::now();

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE];  // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];

  for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;}

  cudaMalloc(&d_A, DSIZE*sizeof(float));  // allocate device space for vector A
  cudaMalloc(&d_B, DSIZE*sizeof(float));  // allocate device space for vector B
  cudaMalloc(&d_C, DSIZE*sizeof(float));  // allocate device space for vector C
  cudaCheckErrors("cudaMalloc failure"); // error checking

  // copy vector A to device:
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  // copy vector B to device:
  cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "Memory setup Time taken: " << duration.count() << " ms" << std::endl;

  //cuda processing sequence step 1 is complete
  int blocks = 1;  // modify this line for experimentation
  int threads = 1; // modify this line for experimentation
// Memory setup Time taken: 15348 ms
// vadd Time taken: 15 ms
// Device to Host Memcpy Time taken: 303693 ms

  //cuda processing sequence step 1 is complete
  blocks = 1;  // modify this line for experimentation
  threads = 4; // modify this line for experimentation
// Memory setup Time taken: 15368 ms
// vadd Time taken: 16 ms
// Device to Host Memcpy Time taken: 91881 ms

  //cuda processing sequence step 1 is complete
  blocks = 512;  // modify this line for experimentation
  threads = 512; // modify this line for experimentation
  // Memory setup Time taken: 15389 ms
  // vadd Time taken: 16 ms
  // Device to Host Memcpy Time taken: 561 ms

  //cuda processing sequence step 1 is complete
  blocks = 512;  // modify this line for experimentation
  threads = 1024; // modify this line for experimentation
  // Memory setup Time taken: 15364 ms
  // vadd Time taken: 18 ms
  // Device to Host Memcpy Time taken: 561 ms

  //cuda processing sequence step 1 is complete
  blocks = 1024;  // modify this line for experimentation
  threads = 1024; // modify this line for experimentation
  // Memory setup Time taken: 15360 ms
  // vadd Time taken: 17 ms
  // Device to Host Memcpy Time taken: 560 ms

    //cuda processing sequence step 1 is complete
  blocks = 1024;  // modify this line for experimentation
  threads = 1; // modify this line for experimentation
  // Memory setup Time taken: 15372 ms
  // vadd Time taken: 15 ms
  // Device to Host Memcpy Time taken: 1527 ms

  start_time = std::chrono::high_resolution_clock::now();

  vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);

  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "vadd Time taken: " << duration.count() << " ms" << std::endl;

  cudaCheckErrors("kernel launch failure");

  start_time = std::chrono::high_resolution_clock::now();

  //cuda processing sequence step 2 is complete
  // copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Device to Host Memcpy Time taken: " << duration.count() << " ms" << std::endl;

  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  return 0;
}
