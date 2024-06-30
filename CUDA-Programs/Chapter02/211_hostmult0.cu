// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.11 hostmult0 simple matrix multiply
//
// RTX 2070
// C:\bin\hostmult0.exe
// A 1024 x 1024 B 1024 x 1024 host time 2301.111 ms Gflops/sec 0.933
// 
// RTX 3080
// C:\bin\hostmult0.exe
// A 1024 x 1024 B 1024 x 1024 host time 2916.772 ms Gflops/sec 0.736

// Line 1: Here we include the thrust host_vector header, std::vector could have been used
// instead for this host only code.
#include "thrust/host_vector.h" // line 1
#include "cxtimers.h"
#include <random>

// Lines 4–12: This is the host matrix multiply function hostmult0, it takes standard pointers to the
// data for the matrices C, A and B as the ﬁrst three arguments. The next three arguments deﬁne the
// sizes of all three matrices. Note we use y and x instead of row and col to denote the ﬁrst and second
// dimensions of the matrices. Thus, A is ay  ax, B is ax  bx and C is ay  bx, we infer the ﬁrst
// dimension of B and both dimensions of C from the properties of matrix multiplication.
int hostmult0(float * C, float * A, float * B, int Ay, int Ax, int Bx) // line 4
{
	// compute C = A * B for matrices (assume Ax = By and C  is Ay x Bx)
	// Lines 7: These for loops over i and j cover all the elements of the desired product C.
	for(int i=0;i<Ay;i++) for(int j=0;j<Bx;j++){ // lines 7

		C[i*Bx+j] = 0.0;      // Cij   = ∑k      Aik  *   Bkj

		// Line 9: The inner loop over k implements the summation from the standard formula. You can think
		// of this summation as a dot product between the ith row of A and the jth column of B. Notice how the
		// array indices vary with the for loop index k. The factor A[i*Ax+k] behaves “nicely” because as
		// k increments, it addresses elements of A which are adjacent in memory, this is optimal for caching.
		// On the other hand, the factor B[k*Bx+j] addresses memory with a stride of Bx words between
		// successive values of k, which gives poor cache performance. This problem is inherent in matrix
		// multiplication and has no simple ﬁx.
		// Notice also that a triple for loop is needed for matrix multiplication. If the matrices have
		// dimensions of 103 then a total of 2  109 arithmetic operations are required – multiplication
		// of big matrices is slow!
		// You might worry that the expressions like i*Bx+j used for the array indices add a signiﬁcant
		// computational load for each step through the loop. In fact, this sort of index expression is so
		// common that compilers are very good at generating the best possible code for indexing such
		// arrays efﬁciently.

		for(int k=0;k<Ax;k++) C[i*Bx+j] += A[i*Ax+k]*B[k*Bx+j]; // line 9

	}

	return 0;
} // line 12


// • Lines 13–36: This is the main routine:
// ○ Lines 15–20: Here we set the matrix sizes using optional user inputs for the dimensions of A
// (Arow & Acol) and the number of columns of B (Bcol). The dimensions of C and number of
// rows of B are set to be compatible with matrix multiplication.
// ○ Lines 21–23: Here we allocate thrust vectors to hold the matrices.
// ○ Lines 25–28: Here A and B are initialised with random numbers.
// ○ Lines 29–31: A timed call the hostmult0 to perform the multiplication.
// ○ Lines 32–35: Print some results, the performance in GFlops/sec assumes two operations per
// iteration in line 9 and ignores all overheads.

int main(int argc,char *argv[])
{
	int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // default 2^10 // line 15
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;

	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;

	int Crow = Arow;
	int Ccol = Bcol; // line 20

	thrust::host_vector<float> A(Arow*Acol); // line 21
	thrust::host_vector<float> B(Brow*Bcol); // line 22
	thrust::host_vector<float> C(Crow*Ccol); // line 23

	// initialise A and B with random numbers
	std::default_random_engine gen(12345678); // line 25
	std::uniform_real_distribution<float> fran(0.0,1.0); // line 26

	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);  // line 27
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);  // line 28

	cx::timer tim; // line 29

	hostmult0(C.data(),A.data(),B.data(),Arow,Acol,Bcol); // line 30

	double t1 = tim.lap_ms(); // line 31

	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol; // line 32
	double gflops= flops/(t1*1000000.0);                       // line 33   
	double gbytes = gflops*6.0; // i.e. 12 bytes per term         line 34

	printf("A %d x %d B %d x %d host time %.3f ms Gflops/sec %.3f\n",
		Arow,Acol,Brow,Bcol,t1,gflops); // line 35

	return 0;
}

// D:\ >hostmult0.exe
// A 1024 x 1024 B 1024 x 1024 host time 2121.046 ms
// GFlops 1.013 GBytes 6.076

// The timing result in the last line shows that this calculation runs at about 1 GFlops/sec and is clearly
// memory bound. The memory bandwidth achieved is about 6 GBytes/sec (8 bytes read and 4 bytes
// written per term).

// The performance of this code is quite poor but we can improve it signiﬁcantly by adding
// the C++11 __restrict keyword to the pointer argument declarations in line 9.
// We do this in the next example, 212_hostmult1.cpp ... 

// A 1024 x 1024 B 1024 x 1024 host time 5742.313 ms Gflops/sec 0.374


