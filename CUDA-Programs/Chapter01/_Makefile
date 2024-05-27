################################################################################
# Simple MakeFile for examples for the book Programming in Parallel with CUDA
# Note customised files for each example are not provided
# please use "make build NAME=gpusum" etc. to provide program names as necessary
#
################################################################################

# default program name. Note use command line option NAME=... as neceessary
NAME := gpusum

# change these to match your installation
CUDA_PATH := /usr/local/cuda
# assumes CUDA samles installed in ~/samples
# NVCUDASAMPLES_ROOT := ~/samples/cuda-samples/Common
NVCUDASAMPLES_ROOT := /home/rob/Data/Documents/Github/NVIDIA/cuda-samples/Common
# assumes examples installed in directory ~/CudaBook 
# CX_ROOT := ~/CudaBook/CUDA-Programs/include
CX_ROOT := /home/rob/Data/Documents/Github/RichardAns/CUDA-Programs/include


# Target rules
all: build

build: $(NAME)

# -ccbin          => Specify the directory in which the default host compiler executable resides.
# g++             => the default host compiler on Linux
# -m64            => Alias for --machine=64
# -threads 0      => Specify the maximum number of threads to be used to execute the compilation steps in parallel.
#                    This option can be used to improve the compilation speed when compiling for multiple architectures. 
#                    The compiler creates number threads to execute the compilation steps in parallel. 
#                    If number is 1, this option is ignored. If number is 0, the number of threads used is the number of CPUs on the machine.
# --use_fast_math => Make use of fast math library.

# 4.2.7.3. --generate-code specification (-gencode) =>
#   ... This option provides a generalization of the --gpu-architecture=arch --gpu-code=code,... option combination for specifying 
#       nvcc behavior with respect to code generation.



$(NAME).o: $(NAME).cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -I $(NVCUDASAMPLES_ROOT)/ -I $(CX_ROOT)/ -g -G -m64  --threads 0 --use_fast_math  -o $(NAME).o -c $(NAME).cu

#$(CUDA_PATH)/bin/nvcc -ccbin g++ -I $(NVCUDASAMPLES_ROOT)/ -I $(CX_ROOT)/ -m64  --threads 0 --use_fast_math -gencode arch=compute_86,code=sm_86 -o $(NAME).o -c $(NAME).cu
    
$(NAME): $(NAME).o   
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64  -o $(NAME) $(NAME).o
#	cp $(NAME) ../../Linux/

# $(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_86,code=sm_86 -o $(NAME) $(NAME).o

run: build
	$(EXEC) ./$(NAME)

testrun: build

clean:
	rm -f    $(NAME) $(NAME).o
	rm -rf ../../Linux/$(NAME)

echo:
	$(info NAME is $(NAME))
#
