// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.8 reduce3
//
// RTX 2070
// C:\bin\reduce3.exe
// sum of 16777216 numbers: host 8388314.9 14.095 ms GPU 8388314.5 0.166 ms
// 
// RTX 3080
// C:\bin\reduce3.exe 24 272 256
// sum of 16777216 numbers: host 8388314.9 15.203 ms GPU 8388314.5 0.112 ms

#include "../include/cx.h"
#include "cxtimers.h"
#include <random>

__global__ void reduce3(float *y,float *x,int N)
{
	extern __shared__ float tsum[];
	int id = threadIdx.x;
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	tsum[id] = 0.0f;
	for(int k=tid;k<N;k+=stride) tsum[id] += x[k];
	__syncthreads();

	// Line 10.1: Here we add a new variable block2 which is set the value of blockDim.x rounded up to
	// the lowest power of 2 greater than or equal to blockDim.x. We use the cx utility function pow2ceil
	// for this. That function is implemented using the NVIDIA intrinsic function __clz(int n) which
	// returns the number of the most signiﬁcant non-zero bit in n. This is a device-only function.
	int block2 = cx::pow2ceil(blockDim.x); // next higher power of 2 ... line 10.1
	
	// Line 10.2: This is the same as line 10 in reduce2 except we use the rounded up block2/2 as the
	// starting value of k.
	for(int k=block2/2; k>0; k >>= 1){     // power of 2 reduction loop  ... line 10.2

		// Line 11: This corresponds to line 11 of reduce2 with an added out-of-range check on id+k.
		if(id<k && id+k < blockDim.x) tsum[id] += tsum[id+k]; // line 11
		__syncthreads();
	}
	if(id==0) y[blockIdx.x] = tsum[0]; // store one value per block
}

int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 256;  // power of 2
	int threads = (argc > 3) ? atoi(argv[3]) : 256;
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000; // set this to 1 for correct answer or >> 1 for timing tests

	thrust::host_vector<float>    x(N);
	thrust::device_vector<float>  dx(N);
	thrust::device_vector<float>  dy(blocks);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	
	for(int k = 0; k<N; k++) x[k] = fran(gen);
	
	dx = x;  // H2D copy (N words)
	
	cx::timer tim;
	
	double host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!
	double t1 = tim.lap_ms();

	// simple GPU reduce for any value of N
	tim.reset();

	double gpu_sum = 0.0;
	for(int rep=0;rep<nreps;rep++){
		reduce3<<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
		reduce3<<<     1, blocks, blocks*sizeof(float)>>>(dx.data().get(),dy.data().get(),blocks);
		if(rep==0) gpu_sum = dx[0];
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms()/nreps;
	
	//double gpu_sum = dx[0];  // D2H copy (1 word)
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",N,host_sum,t1,gpu_sum,t2);

	return 0;
}

// D:\ >reduce3.exe 16777216 288 256
// sum of 16777216 numbers: host 8388314.9 14.012 ms
// GPU 8388314.5 0.196 ms

// In the last line we see that launching this kernel with exactly 8 thread blocks per SM gives a speed-up
// of 2.73 compared to reduce0, slightly better than reduce2.

// "program": "${workspaceFolder}/CUDA-Programs/Chapter02/208_reduce3",
// sum of 16777216 numbers: host 8389645.1 401.875 ms GPU 8389645.0 0.173 ms
// sum of 16777216 numbers: host 8389645.1 411.263 ms GPU 8389645.0 0.198 ms

// The reduce3 kernel is about 70 times faster than the single core host version. While this
// is not quite as spectacular as our Chapter 1 result for a CPU bound calculation, reduction is a
// memory bandwidth bound calculation with just one add per read of 4-bytes of memory so we
// expect reduced performance. Given that the GPU memory bandwidth is only about 10 times
// that of the CPU the factor 70 improvement shows that other GPU features including the
// latency hiding are helping speed up this memory bound problem.


