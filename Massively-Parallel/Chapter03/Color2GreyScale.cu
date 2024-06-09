// Color2GreyScale.cu
// Sunday, June 9, 2024
// Take a color image of any size and convert it to a gray scale image.


#include <stdio.h>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
// #include <opencv2/opencv.hpp>
// #include <jpeglib.h>
// #include <stdexcept>

// Function to check if the file exists
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Function to get image dimensions
// void getImageDimensions(const std::string& filename, int& width, int& height) {
//     cv::Mat image = cv::imread(filename);
//     if (image.empty()) {
//         throw std::runtime_error("Could not open or find the image");
//     }
//     width = image.cols;
//     height = image.rows;
// }

int main(int argc, char * argv[])
{

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return -1;
    }

    std::string filename = argv[1];

    if (!fileExists(filename)) {
        std::cerr << "File does not exist: " << filename << std::endl;
        return -1;
    }
    

    
    




}

