// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// gputiled example 2.16 tiled matrix multiplication on GPU using shared memory.
// 
// RTX 2070
// C:\bin\gputiled.exe 1024 1024 1024 32
// A 1024 x 1024 B 1024 x 1024 gpu time 2.252 ms GFlops 953.460 GBytes 5720.762 (gputiled)
// 
// RTX 3080
// C:\bin\gputiled.exe 1024 1024 1024 32
// A 1024 x 1024 B 1024 x 1024 gpu time 1.125 ms GFlops 1908.968 GBytes 11453.806 (gputiled)

#include "../include/cx.h"
#include "cxtimers.h"
#include <random>

//template <int TS> __global__ void gputiled(float * __restrict C,
//	float * __restrict A,float * __restrict B,int Ay,int Ax,int Bx)

// THIS CODE HAS CHANGED FROM THE ABOVE TO THE BELOW AND DIFFERS FROM WHAT IS SHOWN IN THE BOOK!
// Line 40: This is the start of our new guptiled kernel; the arguments are as before and we are now
// using the restrict keyword by default for all pointers. Note that this is a templated kernel; thus
// the tile size parameter TS is known at compile time.
template <int TS> __global__ void gputiled(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B,int Ay,int Ax,int Bx) // line 40
{
	// Lines 42–43: We declare two statically allocated shared memory arrays to hold square tiles copied
	// from A and B to Atile and Btile.
	__shared__ float Atile[TS][TS];  // tile in A eg [16][16] ... line 42
	__shared__ float Btile[TS][TS];  // tile in B eg [16][16] ... line 43

	// Lines 44–45: Here we set the position of the current thread in the local TS x TS tiles. This depends
	// only on the thread block dimensions.
	int tx  = threadIdx.x;            // tile col index j ... line 44
	int ty  = threadIdx.y;            // tile row index i ... line 45

	// Lines 46–47: Here we set ocx and ocy to the origin of the target tile in C using grid-block
	// quantities. These values are the same for all threads in the thread block.
	int ocx = blockDim.x*blockIdx.x;  // tile x origin in C (all threads)  ... line 46
	int ocy = blockDim.y*blockIdx.y;  // tile y origin in C (all threads)  ... line 47

	// Lines 48–51: In the ﬁrst two lines we set ax and ay to the current thread’s position in A based on the
	// ﬁrst tile to be used. Similarly, in the second pair of lines we set bx and by for matrix B. Notice that
	// as we step to different tiles along the rows of A and down the columns of B ay and bx are constant
	// whereas ax and by change. In fact ay and bx are the i and j values of the cij element being
	// evaluated by the current thread.
	int ax = tx;      // j or x in first tile on A  ... line 48
	int ay = ocy+ty;  // i or y in first tile on A and C ... line 49
	int bx = ocx+tx;  // j or x in first tile on B and C ... line 50
	int by = ty;      // i or y in first tile on B ... line 51

	// Line 51: The local variable csum is used to accumulate the current thread’s cij value; here we set it
	// to zero.
	float csum = 0.0f;

	// Lines 53–61: Each pass through this loop performs matrix multiplication on one pair of tiles from A
	// and B and accumulates the result in csum.
	for(int t=0; t<gridDim.x; t++){ // line 53

		// Lines 54–55: Here we copy the current tiles from A and B to shared memory. Each thread copies
		// one element from A and one from B to Atile and Btile and will later read TS values back
		// from these arrays.
		Atile[ty][tx] = A[ay*Ax+ax];  // copy A tile to shared mem ... line 54
		Btile[ty][tx] = B[by*Bx+bx];  // copy B tile to shared mem ... line 55
		
		// Line 56: An essential syncthreads here; no thread in the block can safely proceed until all the
		// elements of Atile and Btile have been set.
		__syncthreads(); // line 56

		// Line 57: Matrix multiplication of Atile and Btile; each thread computes one element of
		// the product.
		for(int k=0;k<TS;k++) csum += Atile[ty][k]*Btile[k][tx]; // line 57
		
		// Line 58: A second essential syncthreads; no thread can proceed to the next pass through the
		// for loop until all threads have reached this point.
		__syncthreads(); // line 58

		// Lines 59–60: Here we increment ax and by to point to the required position in the next tiles from
		// A and B.
		ax += TS;         // step A tiles along rows of A ... line 59
		by += TS;         // step B tiles down  cols of B ... line 60
	} // line 61

	// Line 62: Here we store the ﬁnal result in C.
	C[ay*Bx+bx] = csum; // store complete result ... line 62
}

int main(int argc,char *argv[])
{
	int Arow = (argc > 1) ? atoi(argv[1]) : 1 << 10; // default 2^10
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;
	uint tilex = (argc > 4) ? atoi(argv[4]) : 32;
	int nacc = (argc > 5) ? atoi(argv[5]) : 100;   // for timing

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

	// Line 28.3: We use tilex to set both dimensions of the 2D thread blocks used to represent tiles.
	// While it is possible to use non-square tiles, that would complicate the kernel code.
	dim3 threads ={tilex,tilex,1}; // force square ... line 28.3
	dim3 blocks ={(Bcol+threads.x-1)/threads.x,(Arow+threads.y-1)/threads.y,1};

	// Lines 29–31: As before this is the timed block that launches a kernel and waits for completion. The
	// kernel launch itself is now changed because the guptiled kernel is written to use the value of
	// tilex as a template parameter. Here we use a 3-way if-else tree to allow values of 32, 16 or 8 for
	// this parameter. The kernel argument list is the same as before.
	cx::timer tim; // line 29 ...
	for(int k=0;k<nacc;k++){
		if(tilex == 8)	     gputiled< 8><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		else if(tilex == 16) gputiled<16><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		else if(tilex == 32) gputiled<32><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
	}
	cudaDeviceSynchronize();
	double t3 = tim.lap_ms()/(double)(nacc); // ... line 31

	C = dev_C; // D2H copy

	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops = flops/(t3*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term

	printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.3f GBytes %.3f (gputiled)\n",Arow,Acol,Brow,Bcol,t3,gflops,gbytes);

	return 0;
}

// D:\ > gputiled.exe 1024 1024 1024 32
// A 1024 x 1024 B 1024 x 1024 gpu time 1.945 ms
// GFlops 1104.284 GBytes 6625.701

// The result in the last line shows that gputiled delivers more than 1 TFlop/sec of processing. A tile
// size of 32  32 works best on the RTX 2070 GPU used for this test.

// A 1024 x 1024 B 1024 x 1024 gpu time 5.069 ms GFlops 423.657 GBytes 2541.943 (gputiled)


// We note that using shared memory as shown in Example 2.16 gives a signiﬁcant
// performance boost of about 250 GFlops/sec amounting to about 1.1 TFlops/sec overall.
// Although not shown here, we did try running this example without using restrict in
// kernel arguments and found only a small drop in performance. This is presumably because
// we now read from A and B fewer times and hence the performance gain from using
// restrict on the pointers to these arguments is less important.