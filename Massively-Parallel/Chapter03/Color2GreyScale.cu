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

#define NUM_CHANNELS 3
#define BLOCK_DIM 16

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

std::string appendToFilename(const std::string& filename, const std::string& append) {
    // Find the position of the last period in the filename
    size_t lastDotPosition = filename.find_last_of(".");
    
    // If there's no period found or it's not a .jpg file, return the original filename
    if (lastDotPosition == std::string::npos || filename.substr(lastDotPosition) != ".jpg") {
        return filename;
    }
    
    // Insert "_2" before the file extension
    std::string newFilename = filename.substr(0, lastDotPosition) + append + filename.substr(lastDotPosition);
    
    return newFilename;
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

__global__
void color_to_grayscale_conversion(unsigned char* in, unsigned char* out, int width, int height){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < 0 || row >= height || col < 0 || col >= width) return;

    int grey_offset = row * width + col;

    int rgb_offset = grey_offset * NUM_CHANNELS;

    unsigned char r = in[rgb_offset + 0];
    unsigned char g = in[rgb_offset + 1];
    unsigned char b = in[rgb_offset + 2];

    out[grey_offset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
}



int main(int argc, char * argv[])
{

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>, <processed_file>" << std::endl;
        return -1;
    }

    // "/home/rob/Data/Documents/Github/rkaunismaa/CUDA/images/Color2GreyScale"
    std::string filename = argv[1];
    // std::string processedFilename = argv[2];

    // std::string filename = "/home/rob/Data/Documents/Github/rkaunismaa/CUDA/images/GregLemond_BernardHinault.jpg";
    // std::string processedFilename = "/home/rob/Data/Documents/Github/rkaunismaa/CUDA/images/GregLemond_BernardHinault_2.jpg";
    std::string processedFilename = appendToFilename(filename, "_2") ;


    // if (!fileExists(filename)) {
    //     std::cerr << "File does not exist: " << filename << std::endl;
    //     return -1;
    // }

    unsigned char* h_input_image, *h_output_image;
    unsigned char* d_input_image, *d_output_image;

    int block_dim, size, image_width, image_height;
    try {
        getImageDimensions(filename, image_width, image_height);
        std::cout << "Image dimensions: " << image_width << "x" << image_height << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    // // Allocate CUDA memory
    // unsigned char* d_image;
    // size_t imageSize = width * height * sizeof(unsigned char);
    // cudaMalloc(&d_image, imageSize);

    block_dim = BLOCK_DIM ;
    //image_width = atoi(argv[2]);
    //image_height = atoi(argv[3]);
    size = image_width * image_height;

    // Allocate memory for the input and output images on host
    h_input_image = (unsigned char*) malloc(NUM_CHANNELS * size * sizeof(unsigned char));
    h_output_image = (unsigned char*) malloc(size * sizeof(unsigned char));
    
    // Allocate memory for the input and output images on device
    cudaMalloc((void**) &d_input_image, NUM_CHANNELS * size * sizeof(unsigned char));
    cudaMalloc((void**) &d_output_image, size * sizeof(unsigned char));

    // Copy the input image to the device
    cudaMemcpy(d_input_image, h_input_image, NUM_CHANNELS * size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Perform the conversion of the image
    dim3 dimBlock(block_dim, block_dim, 1);
    dim3 dimGrid(ceil((float)image_width/block_dim), ceil((float)image_height/block_dim), 1);
    color_to_grayscale_conversion<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, image_width, image_height);

    // Copy the output back to the host
    cudaMemcpy(h_output_image, d_output_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);


    // // Copy image data to CUDA memory
    // cudaMemcpy(d_image, image.data, imageSize, cudaMemcpyHostToDevice);

    // // Launch the CUDA kernel
    // dim3 blockSize(16, 16);
    // dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    // processImageKernel<<<gridSize, blockSize>>>(d_image, width, height);

    // // Copy result back to host
    // cudaMemcpy(image.data, d_image, imageSize, cudaMemcpyDeviceToHost);

    // Save the processed image
    // std::string processedFilename = "/home/rob/Data/Documents/Github/rkaunismaa/CUDA/images/GregLemond_BernardHinault_2.jpg" ;
    // cv::imwrite(processedFilename, image);

    // cv::imwrite(processedFilename, h_output_image);

    // Clean up
    //cudaFree(d_image);

    // Free the device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    // Free the host memory
    free(h_input_image);
    free(h_output_image);


    return 0;


    

    
    




}

