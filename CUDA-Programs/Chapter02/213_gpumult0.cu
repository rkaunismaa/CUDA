// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.13 gpumult0 GPU simple matrix multiply one thread per output element.
// 
// RTX 2070
// C:\bin\gpumult0.exe 1024 1024 1024 32 32
// A 1024 x 1024 B 1024 x 1024 gpu time 6.489 ms GFlops 330.937 GBytes 1985.623
// 
// RTX 3080
// C:\bin\gpumult0.exe 1024 1024 1024 32 8
// A 1024 x 1024 B 1024 x 1024 gpu time 2.984 ms GFlops 719.634 GBytes 4317.804

#include "../include/cx.h"
#include "cxtimers.h"
#include <random>

// Lines 4–11: The GPU kernel gpumult0 replaces the previous hostmult0 function here. The
// kernel is designed to use one thread to calculate one element of the matrix product. The kernel
// expects to be called with a 2D grid of thread blocks with sufﬁcient threads in the x and y dimensions
// to span all the elements of C. As before x is the column index and y is the row index.
__global__ void gpumult0(float * C,const float * A,const float * B,	int Ay,int Ax,int Bx) // line 4
{
	// Lines 6–7: Here we set tx and ty from the built-in variables to determine which element of C this
	// thread will calculate. These lines effectively replace the loops over i and j used in the host version,
	// we can think of the kernel as effectively calculating all the elements of C in parallel.
	int tx = blockIdx.x*blockDim.x + threadIdx.x;  // col index j ... line 6
	int ty = blockIdx.y*blockDim.y + threadIdx.y;  // row index i ... line 7

	// Line 8: This is an out-of-range check on tx and ty. It is necessary because the dimensions of each
	// thread block may have been rounded up.
	if(ty >= Ay || tx >= Bx) return; // line 8

	// Lines 9–10: Here we calculate one element of C using the standard formula. Notice the factor
	// B[k*Bx+tx] in line 10 still uses a memory stride of Bx words on successive passes through the
	// for loop over k. But now in this parallel kernel adjacent threads will use adjacent elements of
	// B because adjacent threads have adjacent values of tx. Thus L1 caching will be efﬁcient for both
	// factors in the multiplication – this is an interesting example of how parallel CUDA code can provide
	// efﬁcient memory access in situations where single threaded code struggles.
	C[ty*Bx+tx] = 0.0; // line 9
	for(int k=0;k<Ax;k++) C[ty*Bx+tx] += A[ty*Bx+k]*B[k*Bx+tx]; // line 10

} // line 11

int main(int argc,char *argv[])
{
	int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // default 2^10
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;

	// Lines 20.1–20.2: We add two additional user settable parameters tilex and tiley which deﬁne
	// the x and y dimensions of the thread blocks used by the kernel launch. These are equivalent to the
	// threads and blocks parameters we use in many 1D examples.
	uint tilex = (argc > 4) ? atoi(argv[4]) : 32;  // thread-block x  ... line 20.1
	uint tiley = (argc > 5) ? atoi(argv[5]) : 8;   // thread-block y  ... line 20.2

	int nacc = (argc > 6) ? atoi(argv[6]) : 100;   // for timing

	thrust::host_vector<float>       A(Arow*Acol);
	thrust::host_vector<float>       B(Brow*Bcol);
	thrust::host_vector<float>       C(Crow*Ccol);

	// Lines 23.1–23.3: Here we allocate device arrays to hold copies of the matrices A, B and C.
	thrust::device_vector<float> dev_C(Crow*Ccol); // line 23.1
	thrust::device_vector<float> dev_A(Arow*Acol); // line 23.2
	thrust::device_vector<float> dev_B(Brow*Bcol); // line 23.3

	// initialise A and B with random numbers
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);

	// Lines 28.1–28.2: Copy A and B to the device.
	dev_A = A;  // H2D copy ... line 28.1
	dev_B = B;  // H2D copy ... line 28.2

	// • Line 28.3: Set threads to a dim3 triad representing a 2D tile on the matrix C.
	dim3 threads ={tilex,tiley,1}; // line 28.3

	// • Line 28.4: Set blocks as a dim3 with x and y dimensions sufﬁcient for the thread block tiles in
	// threads to span the matrix C. Notice the integer rounding up for cases where the dimensions of C are
	// not exact multiples of tilex and tiley. The out-of-range test in line 8 is necessary for cases
	// where rounding up was needed. Rounding up and consequent testing in kernels are very common in
	// CUDA code written to process general cases where not everything is a power of 2.
	dim3 blocks ={(Bcol+threads.x-1)/threads.x,	(Arow+threads.y-1)/threads.y, 1}; // line 28.4

	// • Lines 29–31: This timed loop is similar to that of Example 2.9 but performs a kernel launch instead
	// of a host function call. The use of cudaDeviceSynchronize is necessary for timing purposes.
	cx::timer tim; // line 29 ...
	for(int k=0;k<nacc;k++){ 
		gpumult0<<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
	}
	cudaDeviceSynchronize();  // wait for kernel
	double t2 = tim.lap_ms()/(double)(nacc); // ... line 31

	// Line 31.1: Here we copy the result back to the host. Although C is not used in the code shown here,
	// it would obviously be used in real-world code. Indeed, we have used C to compare the results from
	// the host and GPU versions and ﬁnd the calculated C ij agree to about 6 signiﬁcant ﬁgures.
	C = dev_C;               // D2H copy ... line 31.1

	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops = flops/(t2*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term

	printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.3f GBytes %.3f\n",
		                                    Arow,Acol,Brow,Bcol,t2,gflops,gbytes);

	return 0;

}

// D:\ >gpumult0.exe 1024 1024 1024 32 32
// A 1024 x 1024 B 1024 x 1024 gpu time 6.685 ms
// GFlops 321.233 GBytes 1927.400

// The timing result in the last line shows that there is an impressive speed-up of about 220 times
// compared to the host calculation in Example 2.12.

// A 1024 x 1024 B 1024 x 1024 gpu time 3.966 ms GFlops 541.425 GBytes 3248.551
