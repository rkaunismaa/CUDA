// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// grid3D example 2.3
// 
// RTX 2070
// C:\bin\grid3D.exe 511
// array size   512 x 512 x 256 = 67108864
// thread block  32 x   8 x   2 = 512
// thread  grid  16 x  64 x 128 = 131072
// total number of threads in grid 67108864
// a[1][7][31] = 511 and b[1][7][31] = 22.605309
// rank_in_block = 511 rank_in_grid = 511 rank of block_rank_in_grid = 0
// 
// 
// C:\bin\grid3D.exe 1234567   (thread 135 in block 2411)
// array size   512 x 512 x 256 = 67108864 
// thread block  32 x   8 x   2 = 512
// thread  grid  16 x  64 x 128 = 131072
// total number of threads in grid 67108864
// a[4][180][359] = 1234567 and b[4][180][359] = 1111.110718
// rank_in_block = 135 rank_in_grid = 1234567 rank of block_rank_in_grid = 2411
// 
// RTX 3080
// C:\bin\grid3D.exe 511
// array size   512 x 512 x 256 = 67108864
// thread block  32 x   8 x   2 = 512
// thread  grid  16 x  64 x 128 = 131072
// total number of threads in grid 67108864
// a[1][7][31] = 511 and b[1][7][31] = 22.605309
// rank_in_block = 511 rank_in_grid = 511 rank of block_rank_in_grid = 0
// 
// C:\bin\grid3D.exe 1234567  (thread 135 in block 2411)
// array size   512 x 512 x 256 = 67108864
// thread block  32 x   8 x   2 = 512
// thread  grid  16 x  64 x 128 = 131072
// total number of threads in grid 67108864
// a[4][180][359] = 1234567 and b[4][180][359] = 1111.110718
// rank_in_block = 135 rank_in_grid = 1234567 rank of block_rank_in_grid = 2411

#include "../include/cx.h"

// __device__  int   a[256][512][512];  // file scope
// __device__  float b[256][512][512];  // file scope


// Notice the array dimensions are in order z, y, x going from left to
// right, where memory is allocated so the adjacent x values are adjacent in memory. This is standard in
// C/C++ but opposite to Fortran which uses x, y, z order. Apart from array subscripts we will use
// “natural” x, y, z ordering in our code. This follows CUDA practice where for example a float4
// variable a has members a.x, a.y, a.z, a.w which are ordered from x to w in memory
__device__  int   a[256][512][1024];  // file scope
__device__  float b[256][512][1024];  // file scope


//                         1024    512     256     1234567
__global__ void grid3D(int nx, int ny, int nz, int id)
{

	// grid3D<<<block3d, thread3d>>>(1024, 512, 256, id);

	// int x = blockIdx.x*blockDim.x + threadIdx.x; // find (x,y,z) in
	// int y = blockIdx.y*blockDim.y + threadIdx.y; // in arrays
	// int z = blockIdx.z*blockDim.z + threadIdx.z; // 

	// dim3 block3d(16, 64, 128); // 16*64*128 = 131072
	int gridDimx = gridDim.x; // 16
	int gridDimy = gridDim.y; // 64
	int gridDimz = gridDim.z; // 128

	if (gridDimx != 16 || gridDimy != 64 || gridDimz != 128)
	{
		printf("gridDim KABOOM!");
		return ;
	}

	// dim3 thread3d(32, 8, 2); // 32*8*2    = 512
	int blockDimx = blockDim.x; // 32
	int blockDimy = blockDim.y; // 8
	int blockDimz = blockDim.z; // 2

	if(blockDimx != 32 || blockDimy != 8 || blockDimz != 2)
	{
		printf("blockDim KABOOM!");
		return;     // out of range?
	};

	int x = blockIdx.x*blockDimx + threadIdx.x; // find (x,y,z) in
	int y = blockIdx.y*blockDimy + threadIdx.y; // in arrays
	int z = blockIdx.z*blockDimz + threadIdx.z; // 

	if(x >=nx || y >=ny || z >=nz) return;     // out of range?

	int array_size = nx * ny * nz;
	int block_size = blockDimx * blockDimy * blockDimz;
	int grid_size  = gridDimx * gridDimy * gridDimz;

	int total_threads = block_size * grid_size;

	int thread_rank_in_block = (((threadIdx.z * blockDimy) + threadIdx.y) * blockDimx) + threadIdx.x;
	int block_rank_in_grid  =  (((blockIdx.z * gridDimy) + blockIdx.y) * gridDimx) + blockIdx.x;
	int thread_rank_in_grid = thread_rank_in_block + (block_size * block_rank_in_grid);

	// do some work here
	// ... notice the order of the dimensions! ... it's NOT x,y,z .. !
	a[z][y][x] = thread_rank_in_grid;
	b[z][y][x] = sqrtf((float)a[z][y][x]);

	if(thread_rank_in_grid == id) {

		printf("--- START ---\n");

		printf("array size   %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);

		printf("thread  grid %3d x %3d x %3d = %d\n", gridDimx, gridDimy, gridDimz, grid_size);
		printf("thread block %3d x %3d x %3d = %d\n", blockDimx, blockDimy, blockDimz, block_size);
		
		printf("total number of threads in grid %d\n", total_threads);

		printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n", z, y, x, a[z][y][x], z, y, x, b[z][y][x]);

		printf("rank_in_block = %d rank_in_grid = %d rank of block_rank_in_grid = %d\n", thread_rank_in_block, thread_rank_in_grid, block_rank_in_grid);
		
		printf("--- END ---\n");
	}
}

int main(int argc,char *argv[])
{
	int id = (argc > 1) ? atoi(argv[1]) : 12345;

	dim3 block3d(16, 64, 128); // 16*64*128 = 131072
	dim3 thread3d(32, 8, 2); // 32*8*2    = 512

	// grid3D<<<block3d, thread3d>>>(512, 512, 256, id);
	grid3D<<<block3d, thread3d>>>(1024, 512, 256, id);

    cudaDeviceSynchronize(); // necessary in Linux to see kernel printf

	return 0;
}
