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

	
// Declare two large 3D arrays which have ﬁle scope and so can be used by any of
// the functions declared later in the same ﬁle. This is standard C/C++ but with an extra CUDA
// feature. By declaring the arrays with the __device__ preﬁx we are telling the compiler to allocate
// these arrays in the GPU memory not in the host memory. Thus, the arrays a and b are usable by kernel
// functions but not host functions.
// Notice the array dimensions are in order z, y, x going from left to
// right, where memory is allocated so the adjacent x values are adjacent in memory. This is standard in
// C/C++ but opposite to Fortran which uses x, y, z order. Apart from array subscripts we will use
// “natural” x, y, z ordering in our code. This follows CUDA practice where for example a float4
// variable a has members a.x, a.y, a.z, a.w which are ordered from x to w in memory
__device__  int   a[256][512][1024];  // file scope
__device__  float b[256][512][1024];  // file scope

// The kernel grid3D is declared with four arguments which are the array dimensions and id
// which speciﬁes the thread whose information will be printed
//                         1024    512     256     1234567
// These numbers DO NOT CHANGE FOR THIS KERNEL FUNCTION!
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

	// This is an out-of-range check on the calculated indices. This check is not strictly necessary
	// here as we have carefully crafted the launch parameters to exactly ﬁt the array dimensions. In
	// general, this will not always be possible and it is good practice to always include range checks in
	// kernel code.
	if (gridDimx != 16 || gridDimy != 64 || gridDimz != 128)
	{
		printf("gridDim KABOOM!");
		return ;
	}

	// dim3 thread3d(32, 8, 2); // 32*8*2    = 512
	int blockDimx = blockDim.x; // 32
	int blockDimy = blockDim.y; // 8
	int blockDimz = blockDim.z; // 2

	if(blockDimx != 32 || blockDimy != 8 || blockDimz != 2 || (blockDimx * blockDimy * blockDimz) > 1024)
	{
		printf("blockDim KABOOM!");
		return;     // out of range?
	};

	int x = blockIdx.x*blockDimx + threadIdx.x; // find (x,y,z) in
	int y = blockIdx.y*blockDimy + threadIdx.y; // in arrays
	int z = blockIdx.z*blockDimz + threadIdx.z; // 

	int block_size = blockDimx * blockDimy * blockDimz;
	int grid_size  = gridDimx * gridDimy * gridDimz;

	int total_threads = block_size * grid_size;

	// The rank of the thread within its 3D thread block is calculated using the standard 3D
	// addressing rank formula:
	// rank = (z*dim_y + y)*dim_x + x
	// for a 3D array of dimensions (dim_x, dim_y, dim_z) laid out sequentially in memory with
	// the x values adjacent, the y values are separated by stride of dim_x and the z values are
	// separated by a stride of dim_x*dim_y. We will use versions of this formula very often in our
	// examples, often encapsulated in a lambda function.
	
	int thread_rank_in_block = (((threadIdx.z * blockDimy) + threadIdx.y) * blockDimx) + threadIdx.x;

	// Here we also use the rank formula to calculate the rank of the thread block within the grid
	// of thread blocks.
	int block_rank_in_grid  =  (((blockIdx.z * gridDimy) + blockIdx.y) * gridDimx) + blockIdx.x;
	
	// Here we use the 2D version of the rank formula to calculate the rank of the thread within
	// the entire thread grid
	int thread_rank_in_grid = (block_rank_in_grid * block_size) + thread_rank_in_block;

	// These next 2 lines have nothing to do with the kernel ... 
	if(x >=nx || y >=ny || z >=nz) return;     // out of range?
	int array_size = nx * ny * nz;

	// Notice the array dimensions are in order z, y, x going from left to
	// right, where memory is allocated so the adjacent x values are adjacent in memory. This is standard in
	// C/C++ but opposite to Fortran which uses x, y, z order. Apart from array subscripts we will use
	// “natural” x, y, z ordering in our code. This follows CUDA practice where for example a float4
	// variable a has members a.x, a.y, a.z, a.w which are ordered from x to w in memory
	a[z][y][x] = thread_rank_in_grid;
	b[z][y][x] = sqrtf((float)a[z][y][x]);

	if(thread_rank_in_grid == id) {

		printf("--- START ---\n");

		printf("array size   %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);

		printf("thread  grid %3d x %3d x %3d = %d\n", gridDimx, gridDimy, gridDimz, grid_size);
		printf("thread block %3d x %3d x %3d = %d\n", blockDimx, blockDimy, blockDimz, block_size);
		
		printf("total number of threads in grid %3d x %3d = %d\n", grid_size, block_size, total_threads);

		printf("a[%d][%d][%d] = %i \n", z, y, x, a[z][y][x]);
		printf("b[%d][%d][%d] = %f \n", z, y, x, b[z][y][x]);

		printf("[block_rank_in_grid %d x block_size %d] + thread_rank_in_block %d = thread_rank_in_grid %d\n", block_rank_in_grid, block_size, thread_rank_in_block, thread_rank_in_grid);

		printf("block_rank_in_grid = %d thread_rank_in_block = %d rank of thread_rank_in_grid = %d\n", block_rank_in_grid, thread_rank_in_block, thread_rank_in_grid);
		
		printf("--- END ---\n");
	}
}

int main(int argc,char *argv[])
{
	int id = (argc > 1) ? atoi(argv[1]) : 12345;

	// Here we calculate the thread’s x, y and z coordinates within its thread block. The launch
	// parameters deﬁned in lines 35–36 set the block dimensions to 32, 8 and 2 and the grid dimensions to
	// 16, 64 and 128 for x, y and z respectively. This means that in line 9 the built-in variables
	// blockDim.x and gridDim.x are set to 32 and 16 respectively. Thus threadIdx.x and
	// blockIdx.x will have ranges [0,31] and [0,16] and the desired coordinate x will have the
	// range [0,511] which is required to index the global arrays a and b. Similarly, y and z have ranges
	// of [0,511] and [0,255]. Within any particular thread block the threadIdx values will have
	// ranges of [0,31], [0,7] and [0,1] for x, y and z; note the x range corresponds to one
	// complete warp of threads; this is a design choice not chance. Having decided to use an x range of
	// 32 we are restricted to smaller ranges for y and z as the product of all three is the thread block size
	// which is limited by hardware to a maximum of 1024.

	//             x,  y,   z
	dim3 block3d( 16, 64, 128); // 16*64*128 = 131072 ... line 35
	dim3 thread3d(32,  8,   2); // 32*8*2    = 512 ........ line 36

	// grid3D<<<block3d, thread3d>>>(512, 512, 256, id);
	grid3D<<<block3d, thread3d>>>(1024, 512, 256, id);

    cudaDeviceSynchronize(); // necessary in Linux to see kernel printf

	return 0;
}


// Thursday, June 27, 2024
//   "program": "${workspaceFolder}/CUDA-Programs/Chapter02/203_grid3D",
//   "args": "511" 
// Case id=511: This is the last thread in the ﬁrst block which spans the range: [0-31,0-7,
// 0-1] and the last point in this range is (31,7,1) which is shown correctly as the index
// [1][7][31] in the ﬁgure.
// --- START ---
// array size   1024 x 512 x 256 = 134217728
// thread  grid  16 x  64 x 128 = 131072
// thread block  32 x   8 x   2 = 512
// total number of threads in grid 131072 x 512 = 67108864
// a[1][7][31] = 511 
// b[1][7][31] = 22.605309 
// [block_rank_in_grid 0 x block_size 512] + thread_rank_in_block 511 = thread_rank_in_grid 511
// block_rank_in_grid = 0 thread_rank_in_block = 511 rank of thread_rank_in_grid = 511
// --- END ---

// Thursday, June 27, 2024
//   "program": "${workspaceFolder}/CUDA-Programs/Chapter02/203_grid3D",
//   "args": "1234567" 
// Case id=1234567: To understand this we need to realise that a set of 16 blocks will span
// the complete x range for eight consecutive y and two consecutive z values. Hence the ﬁrst
// 1024 blocks will span the range [0-511,0-511,0-1] which is two complete x-y
// slices of the array, The next 1024 blocks will span the slices with z in range [2-3] and so
// on. Since 1234567 = 512*2411+135 we have picked the 135th thread in the 2412th
// block. The ﬁrst 4 x-y slices account for 2048 blocks so our pick is in the 364th block in the
// 4–5 slice pair. Next since 364 = 22*16 + 12 we conclude that our thread is in the 12th
// block in the set of 16 blocks that spans the index range [0-511,168-175,5-6]. This
// 12th block spans [352-383,176-183,5-6] and since the 135th thread is offset by
// [7,4,0] from this position we ﬁnd an index set of [359,180,5] or a C/C++ 3D
// vector index address of [4][180][359].
// --- START ---
// array size   1024 x 512 x 256 = 134217728
// thread  grid  16 x  64 x 128 = 131072
// thread block  32 x   8 x   2 = 512
// total number of threads in grid 131072 x 512 = 67108864
// a[4][180][359] = 1234567 
// b[4][180][359] = 1111.110718 
// [block_rank_in_grid 2411 x block_size 512] + thread_rank_in_block 135 = thread_rank_in_grid 1234567
// block_rank_in_grid = 2411 thread_rank_in_block = 135 rank of thread_rank_in_grid = 1234567
// --- END ---
// As our second case illustrates 3D thread blocks are somewhat complicated to visualise but
// their unique selling point is that they group threads spanning 3D subregions of the array into
// a single SM unit where the threads can cooperate. In many volume processing applications,
// for example, automatic anatomical segmentation of 3D MRI scans, this is a key advantage.
// In practice, addressing such a subregion directly from the GPU main memory is often
// inefﬁcient due to the large strides between successive y and z values. In such cases caching
// a 3D subregion in shared memory on the SM is an important optimisation.


// Thursday, June 27, 2024
//   "program": "${workspaceFolder}/CUDA-Programs/Chapter02/203_grid3D",
//   "args": "67108863" 
// --- START ---
// array size   1024 x 512 x 256 = 134217728
// thread  grid  16 x  64 x 128 = 131072
// thread block  32 x   8 x   2 = 512
// total number of threads in grid 131072 x 512 = 67108864
// a[255][511][511] = 67108863 
// b[255][511][511] = 8192.000000 
// [block_rank_in_grid 131071 x block_size 512] + thread_rank_in_block 511 = thread_rank_in_grid 67108863
// block_rank_in_grid = 131071 thread_rank_in_block = 511 rank of thread_rank_in_grid = 67108863
// --- END ---


// __device__  int   a[256][512][1024];  // file scope
// __device__  float b[256][512][1024];  // file scope
// Console Output ...
// Passed in 67108864-1=67108863 for the id ... very last thread in the grid! Anything higher than this and we get no output

// --- START ---
// array size   1024 x 512 x 256 = 134217728
// thread  grid  16 x  64 x 128 = 131072
// thread block  32 x   8 x   2 = 512
// total number of threads in grid 131072 x 512 = 67108864
// a[255][511][511] = 67108863 
// b[255][511][511] = 8192.000000 
// [block_rank_in_grid 131071 x block_size 512] + thread_rank_in_block 511 = thread_rank_in_grid 67108863
// block_rank_in_grid = 131071 thread_rank_in_block = 511 rank of thread_rank_in_grid = 67108863
// --- END ---
