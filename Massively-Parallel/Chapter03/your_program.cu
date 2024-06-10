#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>  // For getcwd
#include <cstdio>    // For printf in CUDA

__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {

      // Buffer to hold the current working directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
        return 1;
    }

    // Load an image using OpenCV
    std::string fileName = "/home/rob/Data/Documents/Github/rkaunismaa/CUDA/images/GregLemond_BernardHinault.jpg" ;
    cv::Mat image = cv::imread(fileName);

    if(image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;

    }
    std::cout << "Image dimensions: " << image.cols << " x " << image.rows << std::endl;

    // Launch CUDA kernel
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();

    return 0;
}
