# Manually copied from ..
# ~/Data/Documents/Github/PacktPublishing/Learn-CUDA-Programming/Chapter02/02_memory_overview/03_aos_soa$ 
# into this folder, then tweaked for the solution.cu files ... 
# Also, use as a reference for the remainder of these solution.cu files 

CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=hello_solution matrix_mul_solution vector_add_solution

INCLUDES= -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# my hack ... I always want debugging information.
ALL_CCFLAGS := -g -G

all : ${TARGET}

hello_solution: hello_solution.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

matrix_mul_solution: matrix_mul_solution.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

vector_add_solution: vector_add_solution.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

clean:
	rm -f ${TARGET} 
