// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.7 reduce2
// 
// RTX 2070
// C:\bin\reduce2.exe
// sum of 16777216 numbers: host 8388314.9 14.107 ms GPU 8388314.5 0.166 ms
// 
// RTX 3080
// C:\bin\reduce2.exe 24 256 256
// sum of 16777216 numbers: host 8388314.9 15.662 ms GPU 8388314.5 0.113 ms

#include "../include/cx.h"
#include "cxtimers.h"
#include <random>

// Line 1: This kernel uses y as an output array and x as the input array with N elements. The previous
// reduce1 kernel used x for both input and output.
__global__ void reduce2(float *y,float *x,int N) // line 1
{
	// Line 3: Here we declare the ﬂoat array tsum to be a shared memory array of size determined by the host
	// at kernel launch time. Shared memory is on-chip and very fast. Each SM has its own block of shared
	// memory which has to be shared by all the active thread blocks on that SM. All threads in any given
	// thread block share tsum and can read or write to any of its elements. Inter-block communication is not
	// possible using tsum because each thread block has a separate allocation for its tsum. For this kernel,
	// an array size of blockDim.x is assumed for y and it is up to the host code to ensure that the correct
	// amount has been reserved. Incorrectly speciﬁed kernel launches could cause hard-to-ﬁnd bugs.
	extern __shared__ float tsum[]; // Dynamically Allocated Shared Mem ... line 3

	// Lines 4–6: To prepare for thread-linear addressing we set id to the rank of the current thread in its
	// thread block, tid to the rank of the current thread in the whole grid and stride to the number of
	// threads in the whole grid.
	int id = threadIdx.x; // line 4
	int tid = blockDim.x*blockIdx.x+threadIdx.x; // line 5
	int stride = gridDim.x*blockDim.x; // line 6

	// Line 7: Each thread “owns” one element of tsum, tsum[id] for this part of the calculation. Here
	// we set the element to zero.
	tsum[id] = 0.0f; // line 7

	// Line 8: Here each thread sums the subset of elements of x corresponding to x[id+n*stride] for
	// all valid integers n ≥ 0. Although there is a large stride between successive elements, this is a
	// parallel calculation and adjacent threads will simultaneously be reading adjacent elements of x so
	// this arrangement is maximally efﬁcient for reading GPU main memory. Note that for large arrays,
	// most of the kernel’s execution time is used on this statement and very little calculation is done per
	// memory access.
	for(int k=tid;k<N;k+=stride) tsum[id] += x[k]; // line 8

	// Line 9: The next step of the algorithm requires threads to read elements of tsum that have been
	// updated by different threads in the thread block. Technically that’s ﬁne – this is what shared memory
	// is for. However, not all threads in the thread block run at the same time and we must be sure that all
	// threads in the thread block have completed line 8 before any of the threads proceed. The CUDA
	// function __syncthreads() does exactly this; it acts as a barrier, all (non-exited) threads in the
	// thread block must reach line 9 before any of them can proceed. Note that __syncthreads only
	// synchronises threads in a single thread block. This is in contrast to the host function
	// cudaDeviceSynchronize() which ensures that all pending CUDA kernels and memory transfers
	// have completed before allowing the host to continue. If you want to ensure that all threads in all thread
	// blocks have reached a particular point in a kernel then in most cases your only option is to split the
	// kernel into two separate kernels and use cudaDeviceSynchronize() between their launches.8
	__syncthreads(); // line 9


	// Lines 10–13: This is the implementation of the power of 2 reduction scheme of Figure 2.2 implemented to
	// sum the values in tsum on a thread block. This section of code assumes that blockDim.x is a power
	// of 2. Note that the number of active threads reduces by a factor of 2 on each pass through the for loop.
	// Older tutorials tend to dwell on further optimisation of this loop by explicitly unrolling and exploiting
	// synchronicity within 32-thread warps. This will be discussed in the next chapter on cooperative groups.
	// For now, note further optimisation of this loop is only important for smaller datasets.
	for(int k=blockDim.x/2; k>0; k /= 2){ // power of 2 reduction loop ... line 10
		if(id<k) tsum[id] += tsum[id+k];
		__syncthreads();
	} // line 13

	// Line 14: The ﬁnal block sum accumulated in tsum[0] is stored in the output array y using
	// blockIdx.x as an index.
	if(id==0) y[blockIdx.x] = tsum[0]; // store one value per block ... line 14
}

int main(int argc,char *argv[])
{
	// Lines 18–20: Here we give the user the option to set the array size N and the launch parameters
	// blocks and threads. Note blocks needs to be a power of 2 for the reduce2 kernel to
	// work properly.
	int N       = (argc > 1) ? 1 << atoi(argv[1]) : 1 << 24; // default 2^24 ... line 18
	int blocks  = (argc > 2) ? atoi(argv[2]) : 256;  // power of 2 ... line 19
	int threads = (argc > 3) ? atoi(argv[3]) : 256; // line 20

	int nreps   = (argc > 4) ? atoi(argv[4]) : 1000; // set this to 1 for correct answer or >> 1 for timing tests
	thrust::host_vector<float>    x(N);
	thrust::device_vector<float>  dx(N);
	// • Line 23: We now allocate a device array dy having dimension blocks. This new array will hold
	// the individual block wide reduction sums.
	thrust::device_vector<float>  dy(blocks); // line 23

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

		// Line 35: Here we call the reduce2 kernel for the ﬁrst time to process the whole dx array with the
		// block sums being stored in the output array dy. Note the third kernel argument requesting a shared
		// memory allocation of threads 4-byte ﬂoats for each active thread block. A large value here may
		// result in reduced occupancy.
		reduce2<<<blocks,threads,threads*sizeof(float)>>>(dy.data().get(),dx.data().get(),N); // line 35

		// Line 36: Here we call reduce2 again but with the array arguments swapped round. This has the result
		// of causing the values stored in y by the previous kernel call, to themselves be summed with the total
		// placed in x[0]. This requires a launch conﬁguration of a single thread block of size blocks threads.
		reduce2<<<     1, blocks, blocks*sizeof(float)>>>(dx.data().get(),dy.data().get(),blocks); // line 36

		if(rep==0)  gpu_sum = dx[0];  
	}
	cudaDeviceSynchronize();
	double t2 = tim.lap_ms()/nreps;  // time for one pass here
	//double gpu_sum = dx[0]/nreps;          // D2H copy (1 word) 
	printf("sum of %d numbers: host %.1f %.3f ms GPU %.1f %.3f ms\n",N,host_sum,t1,gpu_sum,t2);
	return 0;
}
// D:\ > reduce2.exe 16777216 256 256
// sum of 16777216 numbers: host 8388314.9 14.012 ms
// GPU 8388314.5 0.202 ms
// The result at the end of the listing shows that reduce2 is about 2.65 times faster than reduce0.

// "program": "${workspaceFolder}/CUDA-Programs/Chapter02/207_reduce2",
// sum of 16777216 numbers: host 8389645.1 423.552 ms GPU 8389645.0 0.214 ms

// A worthwhile optimisation of the reduce2 kernel would be to drop the restriction that blocks
// must be a power of 2. This is because in many GPUs the number of SM units is not a power of 2.
// For example, my GPU has 36 SMs so to keep all SMs equally busy it is better to use 288 rather than
// 256 for the number of user set value of blocks. We can do this by replacing blockDim.x in
// line 10 of the reduce2 kernel by the smallest power of 2 greater than or equal to blocks. For
// blocks = 288 this would be 512. The effect of doing this is that in the ﬁrst pass when k=256,
// threads with rank 0 to 31 will add values from tsum[256] to tsum[287] to their tsum
// values. We also have to add an out-of-range check to prevent threads 32-255 from attempting
// out-of-range additions.

