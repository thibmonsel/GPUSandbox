

#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}
void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++)
    {
        /*
        1. rand(): Generates a pseudo-random integer between 0 and RAND_MAX
        (typically a large value like 32767)
        2. & 0xFF: Performs a bitwise AND with 0xFF (which is 255 in decimal).
        This extracts just the lowest 8 bits (0-255) of the random number,
        effectively giving you a random value between 0 and 255.
        */
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}
int main(int argc, char **argv)
{
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    free(h_A);
    free(h_B);
    free(h_C);

    // Use cudaMalloc to allocate the memory on the GPU
    float *d_A, *d_B, *d_C;
    d_A = cudaMalloc((float **)&d_A, nBytes);
    d_B = cudaMalloc((float **)&d_A, nBytes);
    d_C = cudaMalloc((float **)&d_A, nBytes);

    // Use cudaMemcpy to transfer the data from the host memory to the GPU global memory with the
    // parameter cudaMemcpyHostToDevice specifying the transfer direction.
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);

    // sumArraysOnHost(d_A, d_B, d_C, nElem);
    // When the kernel has fi nished processing all array elements on the GPU, the result is stored on the
    // GPU global memory in array d_C. Copy the result from the GPU memory back to the host array
    // gpuRef using cudaMemcpy. The cudaMemcpy call causes the host to block. The results
    // stored in the array d_C on the GPU are
    // copied to gpuRef as specified with cudaMemcpyDeviceToHost

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // Finally, use cudaFree to release the memory used on the GPU.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /*
    DIFFERENT MEMORY SPACES

    One of the most common mistakes made by those learning to program in CUDA C
    is to improperly dereference the different memory spaces. For the memory allocated
    on the GPU, the device pointers may not be dereferenced in the host code. If you
    improperly use an assignment, for example:
    gpuRef = d_C

    instead of using
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost)
    the application will crash at runtime.

    To help avoid these types of mistakes, Unified Memory was introduced with CUDA
    6, which lets you access both CPU and GPU memory by using a single pointer. You
    will learn more about unified memory in Chapter 4.
    */
    return (0);
}

/*
THREAD HIERARACHY :

This is a two-level thread hierarchy decomposed into blocks of threads and grids of blocks.

(THREAD \in BLOCK \in GRID )

All threads spawned by a single kernel launch are collectively called a grid. All threads in a grid
share the same global memory space. A grid is made up of many thread blocks.
A thread block is a group of threads that can cooperate with each other using:
➤ Block-local synchronization
➤ Block-local shared memory

Threads from different blocks cannot cooperate.

Threads rely on the following two unique coordinates to distinguish themselves from each other:

➤ blockIdx (block index within a grid)
➤ threadIdx (thread index within a block)

CUDA organizes grids and blocks in three dimensions

blockIdx.x
blockIdx.y
blockIdx.z

threadIdx.x
threadIdx.y
threadIdx.z


The dimensions of a grid and a block are specified by the following two built-in variables:

➤ blockDim (block dimension, measured in threads) i.e number of threads per block
➤ gridDim (grid dimension, measured in blocks) i.e number of blocks per grid

blockDim.x
blockDim.y
blockDim.z

GRID AND BLOCK DIMENSIONS

Usually, a grid is organized as a 2D array of blocks, and a block is organized as a
3D array of threads.

Both grids and blocks use the dim3 type with three unsigned integer fields. The
unused fields will be initialized to 1 and ignored.
*/