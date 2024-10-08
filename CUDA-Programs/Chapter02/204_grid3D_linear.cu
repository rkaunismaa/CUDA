// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// grid3D_linear example 2.4
// 
// RTX 2070
// C:\bin\grid3d_linear.exe 1234567 288 256
// array size   512 x 512 x 256 = 67108864
// thread block 256
// thread  grid 288
// total number of threads in grid 73728
// a[4][363][135] = 1234567 and b[4][363][135] = 1111.110718
// rank_in_block = 135 rank_in_grid = 54919 rank of block_rank_in_grid = 214 pass 16
// 
// RTX 3080
// C:\bin\grid3D_linear.exe 1234567 288 256
// array size   512 x 512 x 256 = 67108864
// thread block 256
// thread  grid 288
// total number of threads in grid 73728
// a[4][363][135] = 1234567 and b[4][363][135] = 1111.110718
// rank_in_block = 135 rank_in_grid = 54919 pass 16 tid offset 1179648

#include "../include/cx.h"
#include <locale.h>

// Notice the array dimensions are in order z, y, x going from left to
// right, where memory is allocated so the adjacent x values are adjacent in memory
__device__  int   a[256][512][512];  // file scope
__device__  float b[256][512][512];  // file scope
// __device__  int   a[256][512][1024];  // file scope
// __device__  float b[256][512][1024];  // file scope

__global__ void grid3D_linear(int nx, int ny, int nz, int id)
{

	// setlocale(LC_NUMERIC, "");

	int gridDimx = gridDim.x ;
	int blockDimx = blockDim.x ;

	int tid = (blockIdx.x * blockDimx) + threadIdx.x;

	int array_size = nx * ny * nz;
	int total_threads = (gridDimx * blockDimx) ;

	int tid_start = tid;
	int pass = 0;

	while(tid < array_size){

		// These next 3 lines show how to convert a thread-linear address into 3D coordinates with x the
		// most rapidly varying coordinate and z the least rapidly varying. Note the division and modulus (%)
		// operators are expensive and could be replaced by masking and shifting operations if nx and ny are
		// known powers of 2. This gives better performance at the price of a less general kernel.
		// ---------------------------------
		// Relations Between Linear and 3D indices
		// index = (z*ny+y)*nx+x
		// x = index % nx
		// y = (index / nx) % ny
		// z = index / (nx*ny)
		int x =  tid%nx;        
		int y = (tid/nx)%ny;   
		int z =  tid/(nx*ny); 

		// do some work here
		a[z][y][x] = tid;
		b[z][y][x] = sqrtf((float)a[z][y][x]);

		if(tid == id) {

			printf("--- START ---\n");

			printf("array size   %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);

			printf("grid3D_linear<<<%3d,%3d>>>(%3d,%3d,%3d,%3d);\n", gridDimx, blockDimx, nx, ny, nz, id);
			printf("thread  grid %3d\n", gridDimx);
			printf("thread block %3d\n", blockDimx);

			printf("total number of threads in grid %3d x %3d = %d\n", gridDimx, blockDimx, total_threads);

			printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n",z,y,x,a[z][y][x],z,y,x,b[z][y][x]);
			printf("rank_in_block =  %d rank_in_grid = %d pass %d tid offset %d\n",threadIdx.x,tid_start,pass,tid-tid_start);

			printf("--- END ---\n");
		}

		// Here we increment tid using a stride equal to the length of the entire thread-grid
		tid += gridDim.x*blockDim.x;
		
	// Here we increment a counter pass and continue to the next pass of the while loop. The
	// variable pass is only used as part of the information printed. The actual linear address being used
	// by a given tid within the while loop is rank_in_grid+pass*total_threads.
		pass++;
	}
}

int main(int argc,char *argv[])
{
	int id      = (argc > 1) ? atoi(argv[1]) : 12345;
	int blocks  = (argc > 2) ? atoi(argv[2]) : 288;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;

	grid3D_linear<<<blocks,threads>>>(512,512,256,id);
	//grid3D_linear<<<blocks,threads>>>(1024,512,256,id);

    cudaDeviceSynchronize(); // necessary in Linux to see kernel printf

	return 0;
}

// Thursday, June 27, 2024
//  "program": "${workspaceFolder}/CUDA-Programs/Chapter02/204_grid3D_linear",
//  "args" : "1234567, 288, 256"
// __device__  int   a[256][512][512];  // file scope
// __device__  float b[256][512][512];  // file scope
// grid3D_linear<<<blocks,threads>>>(512,512,256,id);
// --- START ---
// array size   512 x 512 x 256 = 67108864
// grid3D_linear<<<288,256>>>(512,512,256,1234567);
// thread  grid 288
// thread block 256
// total number of threads in grid 288 x 256 = 73728
// a[4][363][135] = 1234567 and b[4][363][135] = 1111.110718
// rank_in_block =  135 rank_in_grid = 54919 pass 16 tid offset 1179648
// --- END ---
// // The results for the thread with a linear index of 1234567, the same value as used in Example 2.2,
// shows that this linear index corresponds to a 3D element [4][363][135] whereas in Example 2.2 using
// 3D grid and thread blocks it corresponded to the element [4][180][359]. Neither result is “wrong”. The
// difference merely reﬂects the different order in which elements of the arrays are encountered.
