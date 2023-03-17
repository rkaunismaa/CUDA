# CUDA
My CUDA learning journey.

## Monday, March 13, 2023

This journey begins with a walk through of the 'CUDA Training Series' by the OAK RIDGE National Laboratory. https://www.olcf.ornl.gov/cuda-training-series/

The repository for this course is found at https://github.com/olcf/cuda-training-series

The relevant code will be pulled from that repository into the CUDA-Training-Series folder of this repository.

All work will be done using Visual Studio Code. MAKE SURE you have the extension 'Nsight Visual Studio Code Edition' installed. It makes all the CUDA programming goodness available to you. 

10:50am The very first time you spin up this folder in Visual Studio Code, it will have no idea of what kind of environment this is (Python? C#? Java?), and so you need to tell it. This is done by creating a launch configuration file, and you can read about this here ...

https://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html

This doc will only work up to a point. You will create a tasks.json file but when you go to Run Build Task, it will fail because there is no Make file. At this point, I add in a Makefile to the work folder, modelled off of ...

https://github.com/PacktPublishing/Learn-CUDA-Programming/blob/master/Chapter01/01_cuda_introduction/02_vector_addition/Makefile

I plan on pulling in resources from this repository in the future. 

The primary take away is USE A MAKEFILE. The stuff in tasks.json just does not work. 

11:46am At this point, I have the code in /CUDA-Training-Series/hw1/matrix_mul_solution.cu running successfully with the cuda debugger. Nice!

Man, I like that! When I run 'git push origin main', I no longer have to submit my credentials every time. Visual Studio Code has saved that for me, and now it just works.

4:54pm Right now I am thinking to go through the CUDA Programming Guide as my primary resource. 

## Tuesday, March 14, 2023

Playing with smokeParticles and Mandelbrot of NVIDIA/cuda-samples/Samples/5_Domain_Specific. Both example folders were copied from /Data/Documents/Github/NVIDIA/cuda-samples/Samples. Dropping into the terminal of either folder and running 'make clean' and then 'make' is able to successfully compile the target example and then run them without problems. However, when I copy them from that folder into this repo, and do the same, it does not work. Running 'make' in either produces the same error:

>>> GCC Version is greater or equal to 5.0.0 <<<
/usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common -m64 --std=c++14 --threads 0 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o GLSLProgram.o -c GLSLProgram.cpp
GLSLProgram.cpp:30:10: fatal error: helper_gl.h: No such file or directory
   30 | #include <helper_gl.h>
      |          ^~~~~~~~~~~~~
compilation terminated.
make: *** [Makefile:369: GLSLProgram.o] Error 255

So what exactly is going on here? Both examples have a ./vscode subfolder. 

Ah, ok. I think I see the problem. Looking at the .vscode/c_cpp_properties.json file of either one, there is a:

 "includePath": [
            "${workspaceFolder}/**",
            "${workspaceFolder}/../../../Common"
        ],

... which of course this repo does not have. So I am going to dump that in here, then try again. 

Ok, nice, that was the problem. They now both compile and run without any problems. 

Hmm I killed the .vscode sub folder in both folders, and I can still compile without any problems. Nice. I also noticed when I removed them, nothing changed when I ran 'git status' and that's probably because the '.' prefix in the name '.vscode' makes the folder hidden. 

10:37am As you continue to work through the NVIDIA cuda-samples, you will learn what to pay attention to as you read the code and run the examples. It's an iterative process that show's you all the CUDA stuff you have no idea about, as well as the quirks when running them in Visual Studio Code on Ubuntu. 

11:17am So I am actualy going to go back over the slides in the CUDA Training Series. 

12:19pm Meh Reading the CUDA C++ Programming Guide.

# Wednesday, March 15, 2023

Going through lesson 3 of the CUDA Training Series. In the video '3) CUDA Optimization (1 of 2)' start listening at round 40 minutes, which explains Launch Configuration. 'GPU Latency Hiding' begins around 51:30. (64 warps per multiprocessor maximum) Each warp is 32 threads. Instructiions are executed in groups of 32 threads, so they are issued warp by warp.

Wow. I have Github CoPilot running and it is amazing what it is able to generate. I am going to have to learn how to use it! 

I think I will opt to pay for it once this trial period is over.

It KNOWS I am writing CUDA code, and it is able to generate CUDA code for me. And it has generated some of these comments for me.! I am stunned at its predictive abilities.

# Thursday, March 16, 2023

Wow, did I ever waste a lot of time today. I was attempting to step through the CUDA-PROGRAMS/Chapter01/gpusum/gpusum.cu program in the "Programming in Parallel with CUDA' book using the repo I pulled down a few days ago. This repo was updated to work with CUDA 12.1. This example uses Thrust, which I have not yet used. After much messing with the makefile, I was able to compile the program, but when I ran it, it failed with the error when it tried to do the thrust call. After much dicking around, I eventuall tried checking out that repo from the branch I created on '2022-06-11' and it worked. So I am going to go back to that branch and start from there. Damnit, I hate it when I waste time like that.

And once again, I am stunned at how good CoPilot is at generating these comments for me. I am going to have to learn how to use it!

8:29pm Hmm still getting some thrust run time issues, and I don't know why. I am going to have to look into this more tomorrow.


