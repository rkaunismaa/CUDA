// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.12 hostmult0  simple matrix multiply with restrict
//
// RTX 2070
// C:\bin\hostmult0.exe
// A 1024 x 1024 B 1024 x 1024 host time 2301.111 ms Gflops/sec 0.933
// 
// RTX 3080
// C:bin\hostmult0.exe
// A 1024 x 1024 B 1024 x 1024 host time 2916.772 ms Gflops/sec 0.736

#include "thrust/host_vector.h"
#include "cxtimers.h"
#include <random>
int hostmult1(float * __restrict C,float * __restrict A,
	float * __restrict B,int Ay,int Ax,int Bx)
{
	// compute C = A * B for matrices (assume Ax = By and C  is Ay x Bx)
	for(int i=0;i<Ay;i++) for(int j=0;j<Bx;j++){
		C[i*Bx+j] = 0.0;      // Cij   = ∑k      Aik  *   Bkj
		for(int k=0;k<Ax;k++) C[i*Bx+j] += A[i*Ax+k]*B[k*Bx+j];
	}
	return 0;
}
int main(int argc,char *argv[])
{
	int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // default 2^10
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;
	thrust::host_vector<float> A(Arow*Acol);
	thrust::host_vector<float> B(Brow*Bcol);
	thrust::host_vector<float> C(Crow*Ccol);
	// initialise A and B with random numbers
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);
	cx::timer tim;
	hostmult1(C.data(),A.data(),B.data(),Arow,Acol,Bcol);
	double t1 = tim.lap_ms();
	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops= flops/(t1*1000000.0);
	double gbytes = gflops*6.0; // i.e. 12 bytes per term
	printf("A %d x %d B %d x %d host time %.3f ms Gflops/sec %.3f\n",
		Arow,Acol,Brow,Bcol,t1,gflops);
	return 0;
}

// D:\ >hostmult0.exe
// A 1024 x 1024 B 1024 x 1024 host time 2121.046 ms
// GFlops 1.013 GBytes 6.076

// D:\ > hostmult1.exe
// A 1024 x 1024 B 1024 x 1024 host time 1468.845 ms
// GFlops 1.462 GBytes 8.772

// We have improved the performance by 44% making this simple change! If you are not
// familiar with the history of C/C++ this requires some explanation. When a function is
// declared with a simple pointer argument, the compiler has no way of being certain that there
// are no other pointers to the same memory location elsewhere in the code. This is called
// pointer aliasing and in the early days of C when computer memory was a scarce resource,
// people would deliberately use pointer aliasing to use the same piece of memory for different
// purposes at different stages in the program. Needless to say, this practice often resulted in
// hard-to-ﬁnd bugs. On modern systems with 64-bit memory addressing pointer aliasing is
// completely unnecessary yet the memory of old practice lingers on in modern compilers
// which are still reluctant to fully optimise code involving simple pointers. Speciﬁcally, they
// will tend to unnecessarily store intermediate results back to main memory rather than using
// registers. Adding the restrict qualiﬁer to a pointer declaration tells the compiler that the
// pointer is not aliased and aggressive optimisation is safe.1



// A 1024 x 1024 B 1024 x 1024 host time 5511.667 ms Gflops/sec 0.390
