# Compiler and linker
NVCC = nvcc
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -g
CUDAFLAGS = -arch=sm_61 -G -g

# OpenCV flags (use pkg-config to get the correct flags)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Include and lib paths for CUDA (optional if nvcc is in PATH)
CUDA_INC = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

# Name of the source file and target (passed via command line)
NAME ?= vector_add

# Source file
SRCS = $(NAME).cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Output binary
TARGET = $(NAME)

# Compile CUDA source file to object file
%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@ $(OPENCV_CFLAGS)

# Link object file to create the executable
$(TARGET): $(OBJS)
	$(NVCC) $(CUDAFLAGS) -o $(TARGET) $(OBJS) $(CXXFLAGS) $(OPENCV_CFLAGS) $(OPENCV_LIBS) -L$(CUDA_LIB) -I$(CUDA_INC)

# Clean
clean:
	rm -f $(TARGET) $(OBJS)