#include <stdio.h>

/*
Comment 2 : __global__ keyword

The qualifier __global__ tells the compiler that the kernel function will 
be called from the CPU and executed on the GPU 
*/

__global__ void helloFromGPU(void)
{
    if(threadIdx.x==5){
    printf("Hello World from GPU thread %u! \n", threadIdx.x);
    }
}
/*
Comment 1 : Compiling and Executing

nvcc hello.cu -o hello 
compile the file hello.cu with nvcc (NVIDIA C Compiler).
run the executable file hello to get Hello World 
*/
int main(void)
{
    printf("Hello World from CPU! \n");
    /*
    Comment 3 : <<<., .>>>
    
    Triple angle brackets mark a call from the host thread to the code on the device side. A kernel is
    executed by an array of threads and all threads run the same code. The parameters within the triple
    angle brackets are the execution configuration, which specifies how many threads will execute the
    kernel. In this example, you will run 10 GPU threads./*

    If we change <<<1, 10>>> to <<<2, 10>>>, 2 CPU threads will be launch and each will spur 10 GPU threads each. 
    */
    helloFromGPU <<<1, 10>>>(); 
    

    /* 
    Comment 4 : cudaDeviceReset()
     
    The function cudaDeviceReset() will explicitly destroy and clean up all resources associated with
    the current device in the current process.
    */
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}

/*
Comment 5 : CUDA PROGRAM STRUCTURE

A typical CUDA program structure consists of five main steps:
1. Allocate GPU memories.
2. Copy data from CPU memory to GPU memory.
3. Invoke the CUDA kernel to perform program-specific computation.
4. Copy data back from GPU memory to CPU memory.
5. Destroy GPU memories.
*/

/*
Comment 6 : 
In the simple program hello.cu, you only see the third step: Invoke the kernel. For
the remainder of this book, examples will demonstrate each step in the CUDA program structure.
*/


/* 
Comment 7 : Locality 

Locality is an important concept in parallel programming. 
If refers to the reuse of data so as to reduce memory access latency.

There are 2 types of locality : 
- temporal locality refers to the reuse of data and/or resources within small time durations.
- spatial locality refers to the reuse of data elements within close storage locations.

CPUs use large cache to optimize application with good spatial and temporal locality. Its the 
programmers responsibility to utilize the CPU cache efficiently.

CUDA exposes you to the concepts of both memory hierarchy and thread hierarchy. 

For example, a special memory, called shared memory, is exposed by the CUDA programming model.
Shared memory can be thought of as a software-managed cache, which provides great speed-
up by conserving bandwidth to main memory. With shared memory, you can control the locality of
your code directly.

When writing a program in CUDA C, you actually just write a piece of serial code to be called by only one thread.
The GPU takes this kernel and makes it parallel by launching thousands of threads, all performing that same computation.
CUDA provides you with a way to organize your threads hierarchically.

CUDA has at its cores 3 key abstractions : 
- a hierarchy of thread groups
- a hierarchy of memory groups
- barrier synchronization
*/


/* 
Comment 8 : CUDA DEVELOPMENT ENVIRONMENT

NVIDIA provides a comprehensive development environment for C and C++ developers to build GPU-accelerated applications, including:

➤ NVIDIA Nsight™ integrated development environment
➤ CUDA-GDB command line debugger
➤ Visual and command line profiler for performance analysis
➤ CUDA-MEMCHECK memory analyzer
➤ GPU device management tools

After you become familiar with these tools, programming with CUDA C is
straightforward and rewarding
*/