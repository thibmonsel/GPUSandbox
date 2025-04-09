/*
A CUDA kernel call is a direct extension to the C function syntax that adds a kernel’s execution
configuration inside triple-angle-brackets:

kernel_name <<<grid, block>>>(argument list);

The first value in the execution configuration is the grid dimension (i.e. number of blocks per grid),
the number of blocks to launch. The second value is the block dimension (i.e. number of threads per blocks),
the number of threads within each block. By specifying the grid and block dimensions, you configure:

➤ The total number of threads for a kernel
➤ The layout of the threads you want to employ for a kernel
*/

/*
Writing your kernel

In a kernel function, you define the computation for a single thread, and the data access for that thread.
When the kernel is called, many different CUDA threads perform the same computation in parallel.

A kernel is defined using the __global__ declaration specification as shown:

__global__ void kernel_name(argument list);

A kernel function must have a void return type

__global__, __device__, __host__ are all function type qualifiers for CUDA.

*/

/*
CUDA KERNELS ARE FUNCTIONS WITH RESTRICTIONS

The following restrictions apply for all kernels:

➤ Access to device memory only
➤ Must have void return type
➤ No support for a variable number of arguments
➤ No support for static variables
➤ No support for function pointers
➤ Exhibit an asynchronous behavior

VERIFYING KERNEL CODE

First, you can use printf in your kernel for Fermi and later generation devices.

Second, you can set the execution configuration to <<<1,1>>>, so you force the
kernel to run with only one block and one thread. This emulates a sequential
implementation. This is useful for debugging and verifying correct results. Also,
this helps you verify that numeric results are bitwise exact from run-to-run if you
encounter order of operations issues.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}
void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    // int i = threadIdx.x;
    // For a more general case if Grid and Block dimension change
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if total number of threads created is larger than the total number of vector
    // elements, you need to restrict your kernel from illegal global memory access
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaSetDevice(dev);
    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);
    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    dim3 block(nElem);
    dim3 grid(nElem / block.x);
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    return (0);
}