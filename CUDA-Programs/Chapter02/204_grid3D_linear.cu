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

// __device__  int   a[256][512][512];  // file scope
// __device__  float b[256][512][512];  // file scope

// Notice the array dimensions are in order z, y, x going from left to
// right, where memory is allocated so the adjacent x values are adjacent in memory
__device__  int   a[256][512][1024];  // file scope
__device__  float b[256][512][1024];  // file scope

__global__ void grid3D_linear(int nx, int ny, int nz, int id)
{

	// setlocale(LC_NUMERIC, "");

	int gridDimx = gridDim.x ;
	int blockDimx = blockDim.x ;

	int tid = (blockIdx.x * blockDimx) + threadIdx.x;

	int array_size = nx * ny * nz;
	int total_threads = (gridDimx * blockDimx;)

	int tid_start = tid;
	int pass = 0;

	while(tid < array_size){

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

		tid += gridDim.x*blockDim.x;
		
		pass++;
	}
}

int main(int argc,char *argv[])
{
	int id      = (argc > 1) ? atoi(argv[1]) : 12345;
	int blocks  = (argc > 2) ? atoi(argv[2]) : 288;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;

	// grid3D_linear<<<blocks,threads>>>(512,512,256,id);
	grid3D_linear<<<blocks,threads>>>(1024,512,256,id);

    cudaDeviceSynchronize(); // necessary in Linux to see kernel printf

	return 0;
}
