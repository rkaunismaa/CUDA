// ChatGPT 3rd kick at the code after telling it the image is not blurred. ...

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) {                                    \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(1);                                                    \
    }                                                               \
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

__global__ void box_blur_kernel(unsigned char *input, unsigned char *output, int width, int height, int channels, int blur_size) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int half_blur_size = blur_size / 2;
        int r = 0, g = 0, b = 0, count = 0;

        for (int ky = -half_blur_size; ky <= half_blur_size; ++ky) {
            for (int kx = -half_blur_size; kx <= half_blur_size; ++kx) {

                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);

                int offset = (ny * width + nx) * channels;
                r += input[offset];
                g += input[offset + 1];
                b += input[offset + 2];
                
                count++;
            }
        }

        int offset = (y * width + x) * channels;
        output[offset] = r / count;
        output[offset + 1] = g / count;
        output[offset + 2] = b / count;
    }
}

void blur_image(const cv::Mat &input, cv::Mat &output, int blur_size) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();

    size_t size = width * height * channels * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    box_blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, blur_size);

    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for any errors launching the kernel
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for the GPU to finish

    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <blur_size>" << std::endl;
        return -1;
    }

    std::string inputFilename = argv[1] ;
    std::string outputFilename = appendToFilename(inputFilename, "_blur") ;

    cv::Mat input = cv::imread(inputFilename, cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    int blur_size = std::stoi(argv[2]);
    if (blur_size <= 0) {
        std::cerr << "Blur size must be greater than 0" << std::endl;
        return -1;
    }

    // Ensure the blur_size is odd
    if (blur_size % 2 == 0) {
        blur_size += 1;
    }

    cv::Mat output(input.rows, input.cols, CV_8UC3);

    blur_image(input, output, blur_size);

    cv::imwrite(outputFilename, output);

    return 0;
}