// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.6 reduce1
// 
// RTX 2070
// C:\bin\reduce1.exe
// reduce1 config 288 256: sum of 16777216 numbers: host 8388314.9 14.261 ms GPU 8388315.5 0.197 ms
// 
// RTX 3080
// C:\bin\reduce1.exe 24 272 256 1000
// reduce1 config 272 256: sum of 16777216 numbers: host 8388314.9 15.546 ms GPU 8388315.0 0.130 ms

#include "../include/cx.h"
#include "cxtimers.h"
#include <random>

// Lines 5–10: This is the reduce1 kernel, now 5 lines long. We use thread-linear addressing to sum all
// the N values contained in x into lower block*threads elements. Each thread accumulates its
// own partial sum in its copy of the register variable tsum and then stores the ﬁnal result in x[tid]
// where tid is the thread’s unique rank in the grid. In this example we have used a for loop instead
// of a while clause to keep the code compact.
// Note line 10, where we change the value of an element of x, requires thought. Not all threads actually
// run at the same time so using the same array for a kernel’s input and output is always potentially
// dangerous. Can we be sure no thread other than tid needs the original value in x[tid]? If the answer
// is no, then the kernel would have a race condition and the results would be undeﬁned. In the present
// case we can be sure because every thread uses a separate disjoint subset of the elements of x. If in doubt
// you should use different arrays for kernel input and output.
__global__ void reduce1(float *x, int N) // line 5
{
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	float tsum = 0.0f;

	int stride = (gridDim.x * blockDim.x);

	for(int k=tid; k<N; k += stride) tsum += x[k];

	x[tid] = tsum; // line 10
}

int main(int argc,char *argv[])
{
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24
	int blocks  = (argc > 2) ? atoi(argv[2]) : 288;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;
	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000;    // set this to 1 for correct answer

	thrust::host_vector<float>    x(N);
	thrust::device_vector<float> dev_x(N);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);

	for(int k = 0; k<N; k++) x[k] = fran(gen);
	//for(int k = 0; k<N; k++) x[k] = 0.5f; // debug accuracy

	dev_x = x;  // H2D copy (N words)

	cx::timer tim;

	double host_sum = 0.0;
	//float host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!

	double t1 = tim.lap_ms();

	// simple GPU reduce for N = power of 2  
	tim.reset();

	double gpu_sum = 0.0;

	// Lines 28–32: Replace the for loop in lines 28–32 of the original. Now we make three calls to
	// reduce1, the ﬁrst uses the full thread-grid deﬁned by the user set variables blocks and
	// threads. After this call the lower blocks*threads elements of x contain partial sums. The second
	// kernel call uses just 1 thread block of size threads, after this call the partial sums are in the ﬁrst
	// threads elements of x. The third call uses just 1 thread to sum threads elements of dx and leave
	// to total sum in the ﬁrst element. Clearly the last two kernel calls do not make efﬁcient use of the GPU.
	// Notice that in the last reduce step, line 31 of Example 2.6, we used a single thread running
	// alone to sum threads values stored in x. We can do better than this by getting the threads
	// in each thread block to cooperate with each other.
	for(int rep=0 ; rep<nreps ; rep++){ // line 28
		reduce1<<< blocks , threads >>>(dev_x.data().get(), N);
		reduce1<<< 1,       threads >>>(dev_x.data().get(), blocks*threads);
		reduce1<<< 1,       1 >>>(      dev_x.data().get(), threads); // line 31
		if (rep==0) gpu_sum = dev_x[0]; // line 32
	}

	cudaDeviceSynchronize();

	double t2 = tim.lap_ms()/nreps;  // time for one pass to compare with host
	//double gpu_sum = dev_x[0];  //D2H copy (1 word) but wrong here for nreps > 1

	printf("reduce1 config %d %d: sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",blocks,threads,N,host_sum,t1,gpu_sum,t2);

	return 0;
}

// From the book ... 
// D:\ > reduce1.exe
// sum of 16777216 numbers: host 8388889.0 14.012 ms
// GPU 8388315.5 0.267 ms

// Saturday, June 29, 2024
// "program": "${workspaceFolder}/CUDA-Programs/Chapter02/206_reduce1", 
// reduce1 config 288 256: sum of 16777216 numbers: host 8389645.1 422.307 ms GPU 8389645.0 0.293 ms

// The bottom line shows the time required for the reduce1 kernel; the host time is unchanged but the
// GPU time is about half that required for reduce0.

// The reduce1 is about twice as fast as reduce0 which is not a bad start but we can do
// more. Our reduce1 kernel is also much more user friendly, it can cope with any value of
// the input array size N and the user is free to tune the launch conﬁguration parameters
// blocks and threads.
