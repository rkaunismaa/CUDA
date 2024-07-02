// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program gpusum_tla includes:
// example 2.1 tls using while loop
// example 2.2 tla using for   loop
// same code with 5th argument to choose method

#include "../include/cx.h"
#include "cxtimers.h"                    // timers

__host__ __device__ inline float sinsum(float x, int terms)
{
	float x2 = x*x;
	float term = x;   // first term of series
	float sum = term; // sum of terms so far

	for(int n = 1; n < terms; n++){
		term *= -x2 / (2*n*(2*n+1));
		sum += term;
	}

	return sum;

}

// __global__ void gpu_sin(float *sums,int steps,int terms,float step_size) // line 15.1
// {
// 	int step = blockIdx.x*blockDim.x + threadIdx.x; // unique thread ID

// 	if(step<steps){  // line 15.4
// 		float x = step_size * step;
// 		sums[step] = sinsum(x, terms);  // store sin values in array ... line 15.6
// 	}
// }

// Our modiﬁcations to the kernel are changing the if in line 15.4 to while and inserting
// an extra line 15.65 at the end of the while loop. In line 15.65 we increment step by the
// total number of threads in the grid of thread blocks. The while loop will continue until steps
// values have been calculated for all (non-zero) user supplied values of blocks and
// threads. Moreover, and importantly for performance reasons, on each pass through the
// while loop adjacent threads always address adjacent memory locations. Other ways of
// traversing through the data could be devised but the one shown here is the simplest and best.
// This technique of using a while loop with indices having a grid-size stride between passes
// through the loop is called “thread-linear addressing” and is common in CUDA code. It
// should always be considered as an option when porting a loop in host code to CUDA.
__global__ void gpu_sin_tla_whileloop(float *sums, int steps, int terms, float step_size)
{
	int step = blockIdx.x*blockDim.x + threadIdx.x; // start with unique thread ID

	while(step<steps){
		float x = step_size*step;
		sums[step] = sinsum(x, terms);  // save sum
		step += blockDim.x*gridDim.x; //  large stride to next step. ... line 15.65
		// blockDim.x = number of threads in one block
		// gridDim.x = number of blocks in the grid
		// threads = gridDim.x*blockDim.x is total number of threads in launch
	}
}

__global__ void gpu_sin_tla_forloop(float *sums, int steps, int terms, float step_size)
{
	for(int step = blockIdx.x*blockDim.x+threadIdx.x; step<steps; step += gridDim.x*blockDim.x){
		float x = step_size*step;
		sums[step] = sinsum(x, terms);  // save sum
	}
}

int main(int argc,char *argv[])
{
	if (argc < 2) {
		printf("usage gpusum_tla steps|10000000 terms|1000 threads|256 blocks|256 loopkind|1\n");
		return 0;
	}
	
	int steps   = (argc > 1) ? atoi(argv[1])  : 10000000;
	int terms   = (argc > 2) ? atoi(argv[2])  : 1000;
	int threads = (argc > 3) ? atoi(argv[3])  : 256;
	int blocks  = (argc > 4) ? atoi(argv[4])  : (steps+threads-1)/threads;  // ensure threads*blocks >= steps
	int loopkind = (argc > 5) ? atoi(argv[5]) : 1;

	double pi = 3.14159265358979323;
	double step_size = pi / (steps-1); // NB n-1 steps between n points

	thrust::device_vector<float> dsums(steps);         // GPU buffer    
	float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer    

	cx::timer tim;                  // declare and start timer

	if (loopkind== 1) {
		gpu_sin_tla_whileloop<<<blocks,threads>>>(dptr,steps,terms,(float)step_size);  // tla using while loop
	}
	else {
		gpu_sin_tla_forloop<<<blocks,threads>>>(dptr,steps,terms,(float)step_size);  // tla using for loop
	}

	double gpu_sum = thrust::reduce(dsums.begin(),dsums.end());

	double gpu_time = tim.lap_ms(); // get elapsed time

	double rate = (double)steps*(double)terms/(gpu_time*1000000.0);
	gpu_sum -= 0.5*(sinsum(0.0f,terms)+sinsum(pi,terms));
	gpu_sum *= step_size;

	if (loopkind==1)printf("gpu_sum while loop sum = %.10f, steps %d terms %d time %.3f ms config %7d %4d rate %f \n",gpu_sum,steps,terms,gpu_time,blocks,threads,rate);
	else            printf("gpu_sum   for loop sum = %.10f, steps %d terms %d time %.3f ms config %7d %4d rate %f \n",gpu_sum,steps,terms,gpu_time,blocks,threads,rate);

	return 0;
}
