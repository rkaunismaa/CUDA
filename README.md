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

