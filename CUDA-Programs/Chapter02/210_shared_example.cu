// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// example 2.10 illustating allocation of 
// multiple dynamic arrays in shared memory
// this example is not intended to be complete.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/cx.h"

// The start of the kernel using three dynamic shared memory arrays is shown in lines 1–12 of this
// example. Here we will assume that the required size of each array is the number of threads in the
// thread block, i.e. threadDim.x for 1D thread grids.
__global__ void shared_example(float *x, float *y, int m) // line 1
{
	// notice order of declarations, 
	// longest  variable type first
	// shortest variable type last
	// In line 6: A single dynamically allocated shared memory array sx of type ﬂoat is declared. Note
	// that sx is just a C style pointer to an array of ﬂoats. We could have used “ﬂoat *sx;” instead of
	// “ﬂoat sx[]”
	extern __shared__ float sx[];   // NB sx is a pointer to the start of the ... line 6

									// shared memory pool

									
	// Lines 7–8: Here pointers to two additional arrays, su and sc, are declared using pointer arithmetic
	// to calculate their offsets from the start of sx. In line 7 the su pointer is set to the address after
	// blockDim.x ﬂoating point elements of the array sx and then cast to the ushort pointer type.
	// Similarly, in line 8 the sc pointer is set to the address after blockDim.x ushort elements of the
	// array su and then cast to the char type.
	ushort* su = (ushort *)(&sx[blockDim.x]); // start after sx ... line 7
	char*   sc =   (char *)(&su[blockDim.x]); // start after su ... line 8
	
	// Lines 9–12: Here we demonstrate use of the arrays, the variable id is set to the current threads’s
	// rank in the thread block and then used normally to index the three arrays.
	int id = threadIdx.x; // line 9
	
	sx[id] = 3.1459*x[id]; // line 10
	su[id] = id*id;        // line 11
	sc[id] = id%128;       // line 12

	// do something useful here . . .
} // line 12

int main(int argc, char * argv[])
{
	// Lines 30–33: These show a fragment of the corresponding host code containing the kernel.
	// ○ Line 30: The launch parameter threads is set using an optional user supplied value.
	// ○ Line 31: The parameter blocks is then set as usual. in lines 30–31.
	// ○ Line 32: A third launch parameter shared is set in line 32. The value stored in shared is
	// calculated as the total number of bytes necessary for the three arrays.
	// ○ Line 33: This shows the kernel launch using three parameters in the launch conﬁguration.

	int threads = (argc >1) ? atoi(argv[1]) : 256;  // line 30
	int size =    (argc >2) ? atoi(argv[2]) : threads*256;  

	int blocks =  (size+threads-1)/threads; // line 31
	int shared = threads*(sizeof(float) + sizeof(ushort) + sizeof(char)); // line 32

	thrust::device_vector<float> x(10);
	thrust::device_vector<float> y(10);
	// do something here

	shared_example<<< blocks, threads, shared >>>(x.data().get(),y.data().get(),10); // line 33
	
	// do more here. . .
	return 0;
}

// One subtle detail of this example is that the calculation made in line 32 makes no
// allowance for memory “gaps” between the arrays that might be needed for natural alignment
// of each array on memory boundaries. However, because the declarations and assignments in
// lines 5–8 of the kernel go from the longest variable type (4-byte ﬂoats) to the shortest
// variable type (1-byte chars), natural alignment will be achieved for all three arrays without
// the compiler needing to introduce gaps.

// Simple variables can also appear in dynamically allocated shared memory, but since
// their size, namely sizeof(variable type), is known at compile time, static
// allocation is the best choice. If the variable is intended to contain some parameter which
// is read but not changed by the threads, then using constant memory might be a
// better choice. Note that constant memory will be automatically used for most kernel
// arguments.
