// All of this code was generated by ChatGPT!
// https://chatgpt.com/c/8c4a2cc1-c320-48ca-821f-e4d602c76ffc
// With the exception of the output file name stuff ... 

// grayscale.cu
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

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


// CUDA kernel to convert RGB to grayscale
__global__ void rgb2gray_kernel(unsigned char *rgb, unsigned char *gray, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int rgb_offset = (y * width + x) * 3;
        int gray_offset = y * width + x;

        unsigned char r = rgb[rgb_offset];
        unsigned char g = rgb[rgb_offset + 1];
        unsigned char b = rgb[rgb_offset + 2];

        gray[gray_offset] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void rgb2gray(const cv::Mat &input, cv::Mat &output) {
    
    int width = input.cols;
    int height = input.rows;

    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);

    unsigned char *d_rgb, *d_gray;
    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_gray, gray_size);

    cudaMemcpy(d_rgb, input.data, rgb_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    rgb2gray_kernel<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);

    cudaMemcpy(output.data, d_gray, gray_size, cudaMemcpyDeviceToHost);

    cudaFree(d_rgb);
    cudaFree(d_gray);
}

int main(int argc, char** argv) {

    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    std::string inputFilename = argv[1] ; 
    std::string outputFilename = appendToFilename(inputFilename, "_gs") ;

    cv::Mat input = cv::imread(inputFilename, cv::IMREAD_COLOR);
    if (input.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }

    cv::Mat output(input.rows, input.cols, CV_8UC1);

    rgb2gray(input, output);

    cv::imwrite(outputFilename, output);

    return 0;
}
