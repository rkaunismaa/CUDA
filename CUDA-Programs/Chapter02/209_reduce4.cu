// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.9 reduce4
// 
// RTX 2070
// C:\bin\reduce4.exe 24 256 256
// sum of 16777216 numbers: host 8388314.9 13.987 ms GPU 8388314.5 0.165 ms
// 
// RTX 3080
// C:\bin\reduce4.exe 24 272 256
// sum of 16777216 numbers: host 8388314.9 15.225 ms GPU 8388314.5 0.113 ms

#include "../include/cx.h"
#include "cxtimers.h"
#include <random>

__global__ void reduce4(float * y,float * x,int N)
{
	extern __shared__ float tsum[];

	int id = threadIdx.x;
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int stride = gridDim.x*blockDim.x;

	tsum[id] = 0.0f;

	for(int k=tid;k<N;k+=stride) tsum[id] += x[k];

	__syncthreads();

	// Lines 10–19: These replace the for loop and last line of the previous example. Here we have unrolled
	// the loop on the explicit assumption that the number of threads per block, blockkDim.x, is in the
	// range [256,511]. In practice we used 256 threads and 288 blocks for the ﬁrst call to reduce4 and
	// 288 threads and 1 block for the second call to reduce4. This kernel could easily be generalised to
	// work with a larger range of thread block sizes, for example, by making the thread block size a template
	// parameter.

	// You can ﬁnd such generalisations in many tutorials; for example, the very early blog by
	// Mark Harris: “Optimizing Parallel Reduction in CUDA, November 2 2007” downloadable from
	// https://developer.download.nvidia.com/assets/cuda/ﬁles/reduction.pdf
	// I have downloaded this file into this folder ... "Optimizing Parallel Reduction in CUDA.pdf"

	// Line 10: This is the ﬁrst step in the parallel reduction chain; values in tsum[256-511] (if any)
	// are added to those in tsum[0-255]. This line is needed for second kernel call with
	// blockDim.x=288. The if statement is then necessary to avoid out of range errors for threads
	// 32-255.
	if(id<256 && id+256 < blockDim.x) tsum[id] += tsum[id+256]; __syncthreads(); // line 10

	// Lines 11–13: These lines are the next three steps in the parallel reduction. No out-of-range checks
	// are needed here on the assumption blockDim.x is at least 256.
	// Note there is a __syncthreads after each step in lines 10–13. These calls are necessary to ensure that
	// all threads in the thread block have completed their addition before any of them proceed to the next step.
	if(id<128) tsum[id] += tsum[id+128]; __syncthreads();  // line 11
	if(id< 64) tsum[id] += tsum[id+ 64]; __syncthreads();  // line 12
	if(id< 32) tsum[id] += tsum[id+ 32]; __syncthreads();  // line 13

	// warp 0 only from here
	// Lines 15–19: These lines are the ﬁnal ﬁve steps in the parallel reduction tree. In these lines only
	// the ﬁrst 32 threads participate. These threads are all in the same warp so we can replace
	// __syncthreads with the much faster __syncwarp. For devices of CC < 7 all threads in the
	// same warp act in strict lockstep so here it is possible to rely on implicit warp synchronisation and
	// omit the __syncwarp calls entirely. You will ﬁnd this done in early (now deprecated) tutorials.
	// Even if you only have access to older devices, we strongly recommend that you always use
	// syncwarp where it would be necessary on newer devices to maintain code portability.
	if(id< 16) tsum[id] += tsum[id+16]; __syncwarp(); // line 15
	if(id< 8)  tsum[id] += tsum[id+ 8]; __syncwarp(); // line 16
	if(id< 4)  tsum[id] += tsum[id+ 4]; __syncwarp(); // line 17
	if(id< 2)  tsum[id] += tsum[id+ 2]; __syncwarp(); // line 18

	if(id==0)  y[blockIdx.x] = tsum[0]+tsum[1]; // line 19

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

		reduce4<<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N);
		reduce4<<<     1,blocks,blocks*sizeof(float)>>>(dx.data().get(),dy.data().get(),blocks);

		if(rep==0) gpu_sum = dx[0];
	}

	cudaDeviceSynchronize();

	double t2 = tim.lap_ms()/nreps;

	//double gpu_sum = dx[0];  // D2H copy (1 word)
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",N,host_sum,t1,gpu_sum,t2);

	return 0;
}

// The performance difference between reduce3 and reduce4 is small but reduce4 has
// introduced us to warp level programming. We will return to the reduce problem in the next
// chapter and show how warp-based programming can be taken much further.

// D:\ >reduce4.exe 16777216 288 256
// sum of 16777216 numbers: host 8388314.9 14.012 ms
// GPU 8388314.5 0.195 ms

// "program": "${workspaceFolder}/CUDA-Programs/Chapter02/209_reduce4",
// sum of 16777216 numbers: host 8389645.1 402.027 ms GPU 8389645.0 0.169 ms


