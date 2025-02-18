#include <cuda_runtime.h>
#include <iostream>

// Function to check for CUDA errors
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    checkCudaError(error, "Failed to get device count");

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, device);
        checkCudaError(error, "Failed to get device properties");

        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Memory pitch: " << deviceProp.memPitch << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: (" << deviceProp.maxThreadsDim[0] << ", "
                                                      << deviceProp.maxThreadsDim[1] << ", "
                                                      << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" << deviceProp.maxGridSize[0] << ", "
                                            << deviceProp.maxGridSize[1] << ", "
                                            << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  Multi-processor count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Maximum memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " << 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
        std::cout << "  Concurrent kernels: " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  ECC enabled: " << (deviceProp.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Unified addressing: " << (deviceProp.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "  Managed memory: " << (deviceProp.managedMemory ? "Yes" : "No") << std::endl;
        std::cout << "  Compute preemption supported: " << (deviceProp.computePreemptionSupported ? "Yes" : "No") << std::endl;
        std::cout << "  Cooperative launch: " << (deviceProp.cooperativeLaunch ? "Yes" : "No") << std::endl;
        std::cout << "  Cooperative multi-device launch: " << (deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No") << std::endl;
    }

    return 0;
}