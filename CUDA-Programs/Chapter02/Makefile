################################################################################
# Simple MakeFile for examples for the book Programming in Parallel with CUDA
# Note customised files for each example are not provided
# please use "make build NAME=gpusum" etc. to provide program names as necessary
#
################################################################################

# default program name. Note use command line option NAME=... as neceessary
NAME := reduce4

# change these to match your installation
CUDA_PATH := /usr/local/cuda
# assumes CUDA samles installed in ~/samples
NVCUDASAMPLES_ROOT := /home/rob/Data/Documents/Github/NVIDIA/cuda-samples/Common
# assumes examples installed in directory ~/CudaBook 
CX_ROOT := /home/rob/Data/Documents/Github/RichardAns/CUDA-Programs/include


# Target rules
all: build

build: $(NAME)

$(NAME).o: $(NAME).cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -I $(NVCUDASAMPLES_ROOT)/ -I $(CX_ROOT)/ -m64 -g -G --threads 0 --use_fast_math -o $(NAME).o -c $(NAME).cu
    
$(NAME): $(NAME).o   
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -g -G -o $(NAME) $(NAME).o

run: build
	$(EXEC) ./$(NAME)

testrun: build

clean:
	rm -f    $(NAME) $(NAME).o

echo:
	$(info NAME is $(NAME))
#
