// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// gputiled1 example 2.17 same as example 2.16 but with one added pragma.
//
// RTX 2070
// C:\bin\gputiled1.exe 1024 1024 1024 32
// A 1024 x 1024 B 1024 x 1024 gpu time 2.093 ms GFlops 1026.142 GBytes 6156.853 (gputiled1)
// 
// RTX 3080
// C:\ bin\gputiled1.exe 1024 1024 1024 32
// A 1024 x 1024 B 1024 x 1024 gpu time 1.049 ms GFlops 2048.090 GBytes 12288.539 (gputiled1)

#include "../include/cx.h"
#include "cxtimers.h"
#include <random>

//template <int TS> __global__ void gputiled(float * __restrict C,
//	float * __restrict A,float * __restrict B,int Ay,int Ax,int Bx)

template <int TS> __global__ void gputiled(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B,int Ay,int Ax,int Bx)
{
	__shared__ float Atile[TS][TS];  // tile in A eg [16][16]
	__shared__ float Btile[TS][TS];  // tile in B eg [16][16]

	int tx  = threadIdx.x;            // tile col index j
	int ty  = threadIdx.y;            // tile row index i
	int ocx = blockDim.x*blockIdx.x;  // tile x origin in C (all threads)    
	int ocy = blockDim.y*blockIdx.y;  // tile y origin in C (all threads)

	int ax = tx;      // j or x in first tile on A
	int ay = ocy+ty;  // i or y in first tile on A and C
	int bx = ocx+tx;  // j or x in first tile on B and C
	int by = ty;      // i or y in first tile on B

	float csum = 0.0f;
#pragma unroll 16
	for(int t=0; t<gridDim.x; t++){
		Atile[ty][tx] = A[ay*Ax+ax];  // copy A tile to shared mem
		Btile[ty][tx] = B[by*Bx+bx];  // copy B tile to shared mem
		__syncthreads();
		for(int k=0;k<TS;k++) csum += Atile[ty][k]*Btile[k][tx];
		__syncthreads();
		ax += TS;         // step A tiles along rows of A
		by += TS;         // step B tiles down  cols of B
	}
	C[ay*Bx+bx] = csum; // store complete result
}

int main(int argc, char *argv[])
{
	int Arow = (argc > 1) ? atoi(argv[1]) : 1 << 10; // default 2^10
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;
	uint tilex = (argc > 4) ? atoi(argv[4]) : 32;
	int nacc = (argc > 5) ? atoi(argv[5]) : 100;   // for timing

	//int Abytes = Arow*Acol;
	int Abytes = 10000000;
	thrust::host_vector<float>       A1(Abytes);
	thrust::device_vector<float> dev_A1(Abytes);

	thrust::host_vector<float>       A(Arow*Acol);
	thrust::host_vector<float>       B(Brow*Bcol);
	thrust::host_vector<float>       C(Crow*Ccol);
	thrust::device_vector<float> dev_A(Arow*Acol);
	thrust::device_vector<float> dev_B(Brow*Bcol);
	thrust::device_vector<float> dev_C(Crow*Ccol);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);
	dev_A = A;  // H2D copy
	dev_B = B;  // H2D copy

	dim3 threads ={tilex,tilex,1}; // force square
	dim3 blocks ={(Bcol+threads.x-1)/threads.x,(Arow+threads.y-1)/threads.y,1};

	cx::timer tim;
	for(int k=0;k<nacc;k++){
		if(tilex == 8)	     gputiled< 8><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		else if(tilex == 16) gputiled<16><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		else if(tilex == 32) gputiled<32><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
	}
	cudaDeviceSynchronize();
	double t3 = tim.lap_ms()/(double)(nacc);
	C = dev_C; // D2H copy

	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops = flops/(t3*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term
	printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.3f GBytes %.3f (gputiled1)\n",Arow,Acol,Brow,Bcol,t3,gflops,gbytes);

	return 0;
}

// D:\ gputiled1.exe 1024 1024 1024 32
// A 1024 x 1024 B 1024 x 1024 gpu time 1.765 ms
// GFlops 1216.958 GBytes 7301.748

// As shown in the last line of Example 2.17 we have gained another 100 GFlops/sec of
// performance by using loop unrolling. The optimal depth of unrolling can only be found by
// experiment; on our RTX 2070 the value 16 seems to give the best result. On other GPUs you
// may ﬁnd a different optimum. Tuning GPU code always involves some experimentation.
// Note the NVCC compiler will often automatically perform loop unrolling and especially in
// cases where the number of passes is known at compile time. For this reason, making the loop
// counter a template parameter can be worthwhile. Here this is done for the inner loop over TS
// but not for the outer loop over gridDim.x which is therefore not known at compile time.
// Interestingly, we ﬁnd that explicit unrolling over the outer loop helps but in experiments (not
// shown) we found explicit unrolling over the inner loop does not help.

// A 1024 x 1024 B 1024 x 1024 gpu time 5.041 ms GFlops 425.966 GBytes 2555.798 (gputiled1)