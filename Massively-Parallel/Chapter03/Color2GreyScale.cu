// Color2GreyScale.cu
// Sunday, June 9, 2024
// Take a color image of any size and convert it to a gray scale image.


#include <stdio.h>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
// #include <opencv2/opencv.hpp>
// #include <jpeglib.h>
// #include <stdexcept>

// Function to check if the file exists
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Function to get image dimensions
void getImageDimensions(const std::string& filename, int& width, int& height) {
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        throw std::runtime_error("Could not open or find the image");
    }
    width = image.cols;
    height = image.rows;
}

// CUDA kernel (simple example)
__global__ void processImageKernel(unsigned char* d_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        // Example: Invert the color
        int index = y * width + x;
        d_image[index] = 255 - d_image[index];
    }
}

int main(int argc, char * argv[])
{

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>, <processed_file>" << std::endl;
        return -1;
    }

    std::string filename = argv[1];
    //std::string processedFilename = argv[2];

    // if (!fileExists(filename)) {
    //     std::cerr << "File does not exist: " << filename << std::endl;
    //     return -1;
    // }

    int width, height;
    try {
        getImageDimensions(filename, width, height);
        std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

     cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    // Allocate CUDA memory
    unsigned char* d_image;
    size_t imageSize = width * height * sizeof(unsigned char);
    cudaMalloc(&d_image, imageSize);

    // Copy image data to CUDA memory
    cudaMemcpy(d_image, image.data, imageSize, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    processImageKernel<<<gridSize, blockSize>>>(d_image, width, height);

    // Copy result back to host
    cudaMemcpy(image.data, d_image, imageSize, cudaMemcpyDeviceToHost);

    // Save the processed image
    std::string processedFilename = "/home/rob/Data/Documents/Github/rkaunismaa/CUDA/images/GregLemond_BernardHinault_2.jpg" ;
    cv::imwrite(processedFilename, image);

    // Clean up
    cudaFree(d_image);

    return 0;


    

    
    




}

