// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.5 reduce0
// 
// RTX 2070
// C:\bin\reduce0.exe
// sum of 16777216 random numbers: host 8388314.9 14.103 ms, GPU 8388315.0 0.643 ms
// 
// RTX 3080
// C:\bin\reduce0.exe
// sum of 16777216 random numbers: host 8388314.9 15.595 ms, GPU 8388315.0 0.569 ms


// Reduce itself involves ﬁnding the arithmetic sum
// of the numbers, but other operations such as max or min would require similar code.
// As a speciﬁc case, consider the problem of summing N ﬂoating point numbers stored in
// the GPUs global memory. The ﬁrst point to recognise is that each data item just requires a
// single add; thus we will be limited by memory access speed not arithmetic performance. This
// is the exact opposite to the situation in Example 1.3. We want to use as many threads as
// possible in order to hide memory latency efﬁciency so our basic algorithm is as shown in the
// box and illustrated in Figure 2.2.

// Reduce Algorithm 1: Parallel sum of N numbers
// • Use N/2 threads to get N/2 pairwise sums
// • Set N = N/2 and iterate till N=1


#include "../include/cx.h"
#include "cxtimers.h"
#include <random>


// Lines 4–8: Show the reduce0 kernel which is very simple; each thread ﬁnds its rank, tid, in the
// grid and, making the tacit assumption that tid is in the range 0 to m-1, adds the appropriate
// element from the top half of the array to the bottom half.
// At this point, it is worth pausing to admire
// the simplicity of the kernel code. We have been able to directly express the idea for implementing a
// parallel reduction with a 2-line kernel.
__global__ void reduce0(float *x, int m) // line 4 ...
{
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	// Another thing to think about when looking at kernel code is the sheer power of the GPU ... 
	// line 7 which does the additions will be executed in parallel by all the cores on the GPU, potentially
	// delivering one or more operations for each core on each clock-cycle.
	x[tid] += x[tid+m];  // line 7

	// the only calculation done by each thread is the above single addition operation. 
	// This line triggers three global memory operations, namely loading both the values stored in x[tid]
	// and x[tid+m] into GPU registers and then storing the sum of these values back into
	// x[tid]. If we could accumulate partial sums in local registers, that would reduce the
	// number of global memory accesses needed for each addition down to one, which offers a
	// speed-up by a potential factor of three.

	// Secondly, the host calls the kernel iteratively, halving the array size at each step to
	// complete the reduction process, leaving the ﬁnal sum in the ﬁrst array element. The effect
	// of this is to double the number of times the x[tid] += x[tid+m] statement is
	// performed. If we could instead perform the iteration inside the kernel that could also reduce
	// the number of memory accesses required.

} // line 8 ...

int main(int argc,char *argv[])
{
	// Line 11: Here we set the array size N to a user supplied value or a default of 2 raised to 24 = 16,777,216
	int N = (argc> 1) ? atoi(argv[1]) : 1 << 24; // default 224 ... line 11

	// N => 16,777,216
	// Lines 12–13: Here we allocate thrust host and device vectors x and dev_x to hold the data.
	thrust::host_vector<float>       x(N); // line 12
	thrust::device_vector<float> dev_x(N); // line 13

	// initialise x with random numbers and copy to dx ... 
	// Lines 15–17: These lines initialise a C++ random number generator and use it to ﬁll x. The use of
	// generators from <random> is much preferred over the deprecated rand() function from ancient C.
	std::default_random_engine gen(12345678); // line 15
	std::uniform_real_distribution<float> fran(0.0, 1.0); // line 16

	for(int k = 0; k<N; k++) x[k] = fran(gen); // line 17

	// Line 18: The contents of x are copied from the host to dev_x on the GPU. The details of the
	// transfer are handled by thrust.
	dev_x = x;  // H2D copy (N words) // line 18

	// Lines 19–22: A timed loop to perform the reduction on the host using a simple for loop.
	cx::timer tim; // line 19

	double host_sum = 0.0;
	for(int k = 0; k<N; k++) host_sum += x[k]; // host reduce!

	double t1 = tim.lap_ms(); // line 22

	int kernelLaunchCount = 0 ;

	// simple GPU reduce for N = power of 2  
	// Lines 24–31: Implement the GPU-based parallel iteration of Algorithm 1. For each pass through the
	// for loop the reduce0 kernel called in line 28 causes the top half of the array dev_x to be
	// “folded” down to an array of size m by adding the top m elements to the bottom m elements. The last
	// pass through the loop has m=1 and leaves the ﬁnal sum in dev_x[0]; this value is copied back to
	// the host in line 32. Lines 28–29: Within the for loop the kernel launch parameters blocks and
	// threads are set so that the total number of threads in the grid is exactly m. This code will fail if
	// N is not a power of 2 due to rounding down errors at one or more steps in the process.
	tim.reset(); // line 24
	for(int m = N/2; m>0; m /= 2) { 

		int blocks =  std::max(m/256, 1);
		int threads = std::min(256, m);

		reduce0<<<blocks,threads>>>( dev_x.data().get(), m); // line 28

		kernelLaunchCount++ ;

	} // line 29
	cudaDeviceSynchronize();  // line 30
	double t2 = tim.lap_ms(); // line 31

	// In CUDA programs a kernel launch such as that used in line 28 will not block the host which will
	// proceed to the next line of the host program without waiting for the kernel call to ﬁnish. In this case
	// that means all the kernel calls (23 in all for N=2^24) will be rapidly queued to run successively on the
	// GPU. In principle the host can do other CPU work while these kernels are running on the GPU. In this
	// case we just want to measure the duration of the reduction operation so before making the time
	// measurement we must use a cudaDeviceSynchronize call in line 30 which causes the host to
	// wait for all pending GPU operations to complete before continuing. This kind of synchronisation issue
	// often occurs in parallel code.

	// Lines 32–33: Here we copy the ﬁnal sum in the dev_x[0] back to the host, again using thrust, and
	// print results.
	double gpu_sum = dev_x[0];  // D2H copy (1 word) // line 32
	printf("Kernel Launched %d times, sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", kernelLaunchCount, N, host_sum, t1, gpu_sum, t2); // line 33

	// The bottom line shows the results obtained running this program with the default value of 2^24 for the
	// number of values to be summed. Note the kernel execution time of 0.535 ms is too short a single
	// measurement to be reliable. The values shown in these reduce examples were in fact obtained as
	// averages of 10,000 runs using a for loop around kernel calls. An alternative method would be to use
	// the Nsight Compute proﬁling tool, but our simple host-based method using cx::timer is a good
	// starting point.

	return 0;
}

// Book values ...
// D:\ > reduce0.exe
// sum of 16777216 random numbers: host 8388314.9 14.012 ms
// GPU 8388315.0 0.535 ms


// Wednesday, July 3, 3034
// run from the terminal, not the debugger ... 
// Kernel Launched 24 times, sum of 16777216 random numbers: host 8389645.1 410.048 ms, GPU 8389646.0 0.242 ms
// launch in VSCode debug mode ... WAYYY longer to run! ... 37.162 / 0.242 = 153.56 
// Kernel Launched 24 times, sum of 16777216 random numbers: host 8389645.1 411.732 ms, GPU 8389646.0 37.162 ms

// Friday, June 28, 2024
// My values ...
//  "program": "${workspaceFolder}/CUDA-Programs/Chapter02/205_reduce0", 
// sum of 16777216 random numbers: host 8389645.1 415.011 ms, GPU 8389646.0 17.529 ms
// sum of 16777216 random numbers: host 8389645.1 423.265 ms, GPU 8389646.0 24.017 ms ... ran this on Saturday, June 29, 2024 ... LMStudio was running
