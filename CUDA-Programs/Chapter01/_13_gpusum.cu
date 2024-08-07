﻿﻿// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program 1.3 gpusum
// 
// RTX 2070
// C:\Users\Richard\OneDrive\toGit2>bin\gpusum.exe 1000000000 1000
// gpu sum = 2.0000000134, steps 1000000000 terms 1000 time 1881.113 ms
// 
// RTX 3080
// C:\Users\Richard\OneDrive\toGit2>bin\gpusum.exe 1000000000 1000
// gpu sum = 1.9999998123, steps 1000000000 terms 1000 time 726.253 ms

#include "../include/cx.h"
#include "cxtimers.h"              // cx timers

__host__ __device__ inline float sinsum(float x,int terms)
{
	float x2 = x*x;
	float term = x;   // first term of series
	float sum = term; // sum of terms so far
	for(int n = 1; n < terms; n++){
		term *= -x2 / (2*n*(2*n+1));  // build factorial
		sum += term;
	}
	return sum;
}

__global__ void gpu_sin(float *sums,int steps,int terms,float step_size)
{
	
	// Line 15.3 contains the magic formula used by each particular
	// instance of an executing thread to ﬁgure out which particular value of the index step that it
	// needs to use.
	int step = blockIdx.x*blockDim.x+threadIdx.x; // unique thread ID

	if(step<steps){
		float x = step_size*step;
		sums[step] = sinsum(x,terms);  // store sin values in array
	}
}

int main(int argc,char *argv[])
{
	int steps = (argc > 1) ? atoi(argv[1]) : 10000000; // get command
	int terms = (argc > 2) ? atoi(argv[2]) : 1000;     // line arguments
	int threads = 256;
	int blocks = (steps+threads-1)/threads;  // ensure threads*blocks ≥ steps

	double pi = 3.14159265358979323;
	double step_size = pi / (steps-1); // NB n-1 steps between n points

	// This line creates the array dsums of size steps in the device memory using
	// the thrust device_vector class as a container. By default the array will be
	// initialised to zeros on the device. This array is used by the gpu_sin kernel
	// to hold individual values returned by calls to the sinum function. 
	thrust::device_vector<float> dsums(steps);         // GPU buffer 

	// We cannot pass dsums to the kernel directly as thrust was not designed
	// to make this possible,4 but we can pass a pointer to the memory array managed by the
	// class.
	float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer

	cx::timer tim;
	// blocks => number of blocks, threads => number of threads per block (1024 max)
	// blocks*threads => total number of threads
	// (threads should be a multiple of warp size 32)
	gpu_sin<<<blocks,threads>>>(dptr,steps,terms,(float)step_size);

	// Here we use the host callable reduce function in the thrust library to sum all
    // the elements of the array dsums in GPU memory. This call involves two steps, ﬁrstly we
    // perform the required additions on the GPU and secondly we copy the result from GPU
    // memory to CPU memory. This is often referred to as a D2H (device to host) transfer.
	double gpu_sum = thrust::reduce(dsums.begin(),dsums.end());

	double gpu_time = tim.lap_ms(); // get elapsed time

	// Trapezoidal Rule Correction
	gpu_sum -= 0.5*(sinsum(0.0f,terms)+sinsum(pi,terms));
	gpu_sum *= step_size;
	printf("gpu sum = %.10f, steps %d terms %d time %.3f ms\n",
		gpu_sum,steps,terms,gpu_time);
	return 0;
}
