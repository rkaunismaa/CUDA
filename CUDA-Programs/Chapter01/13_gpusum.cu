// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
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

#include "cx.h"
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

__global__ void gpu_sin(float *sums,int steps,int terms,float step_size) // line 15.1
{
	int step = blockIdx.x*blockDim.x + threadIdx.x; // unique thread ID

	// Line 15.4: This is an out-of-range check on the value of step, the kernel will exit at this
	// point for threads that fail the check.
	if(step<steps){ // line 15.4
		// Line 15.5: Calculate the x value corresponding to step.
		float x = step_size * step; // line 15.5
		// Line 15.6: Call sinsum with the thread dependant value of x. The result is stored in the
		//array sums using step as an index.
		sums[step] = sinsum(x, terms);  // store sin values in array ... line 15.6
	}
} // Line 15.7: The kernel exits at here; recall that return statements are not required for void functions in C++.

// The kernel declaration in line 15.1 looks very much like a normal C++ declaration
// except for the preﬁx __global__. There are, however, some restrictions based on the fact
// that although the kernel is called from the host it cannot access any memory on the host. All
// kernels must be declared void and their arguments are restricted to scalar items or pointers
// to previously allocated regions of device memory. All kernel arguments are passed by value.
// In particular, references are not allowed. It is not a good idea to try and pass large C++
// objects to kernels; this is because they will be passed by value and there may be signiﬁcant
// copying overheads. Also any changes made by the kernel will not be reﬂected back in the
// host’s copy after the kernel call. Additionally, any C++ classes or structs passed to a kernel
// must have __device__ versions of all their member functions.

// There is another important point to make about line 15.1. The GPU hardware allocates
// all the threads in any particular thread block to a single SM unit on the GPU, and these
// threads are run together very tightly on warp-engines as warps of 32 threads. The variable
// threadIdx.x is set so that threads in the same warp have consecutive values of
// this variable; speciﬁcally threadIdx.x%32 is the rank or lane of a thread within its
// warp (range 0–31) and threadIdx.x/32 is the rank of the warp within the thread
// block (range 0–7 in our case). Thus, in line 15.6 of the kernel where we store a value in
// sums[step], the adjacent threads within a given warp have adjacent values of step and
// so they will address adjacent memory locations in the array sums.



int main(int argc,char *argv[])
{
	int steps = (argc > 1) ? atoi(argv[1]) : 10000000; // get command
	int terms = (argc > 2) ? atoi(argv[2]) : 1000;     // line arguments

	// Lines 19.1–19.2: The two lines are added to deﬁne the kernel launch conﬁguration
	// parameters threads and blocks. In this our ﬁrst example, we use a ﬁxed value
	// of 256 for threads and a calculated value for blocks which is set to be just big enough
	// to get the total number of threads in the launch to satisfy threads × blocks ≥
	// steps.

	// threads should be a multiple of
	// 32 and has a maximum allowed value of 1024 for all current GPUs
	int threads = 256; // line 19.1

	// For most kernels a good starting point for blocks is <<<4*Nsm, 256>>> where Nsm is the number of SMs on
	// the target GPU.6
	// 4090 has 128 Multiprocessors ... 4x128=512
	int blocks = (steps+threads-1)/threads;  // ensure threads*blocks ≥ steps ... line 19.2
	// for this example, blocks = 39063, (10000000 + 256 -1)/ 256 => 39,063

	double pi = 3.14159265358979323;
	double step_size = pi / (steps-1); // NB n-1 steps between n points

	// This next line creates the array dsums of size steps in the device memory using
	// the thrust device_vector class as a container. By default the array will be initialised to
	// zeros on the device. This array is used by the gpu_sin kernel to hold the individual
	// values returned by calls to the sinsum function.
	thrust::device_vector<float> dsums(steps);         // GPU buffer 

	// We cannot pass dsums to the kernel directly as thrust was not designed
	// to make this possible, but we can pass a pointer to the memory array managed by the
	// class. For std::vector objects, the member function data() does this job. While
	// this function does work for thrust host_vector objects it does not work for
	// device_vector objects. Therefore we have to use the more complicated cast shown
	// in this line. As an alternative you could instead use the undocumented data().get()
	// member function of device_vectors.
	float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer

	cx::timer tim;

	gpu_sin<<<blocks,threads>>>(dptr, steps, terms, (float)step_size);

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
