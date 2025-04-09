#include <stdio.h>
#include "../day_common/day_common.h"
#include <cuda_runtime.h>

/*
An example of a vector addition on the host and device. Using a simple kernel to add two
vectors together. The kernel is launched with a grid of blocks, where each block contains a
number of threads. Each thread is responsible for adding one element of the two vectors
together.
*/

void initializeData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void vectorAddOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
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
        printf("Arrays match.\n");
}

__global__ void vectorAddOnDevice(float *A, float *B, float *C, const int N)
{
    // Adding the keyword __global__ means that the CPU host will launch this
    // kernel on the device.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    printf("Starting... \n");
    // Setting size of my arrays
    // N = 1M elements
    const int N = 1 << 20;
    // site_t is a unsigned int
    size_t nBytes = N * sizeof(float);
    printf("Vector size: %d that will be of size %d bytes (sizeof(float)=%d) \n", N, nBytes, sizeof(float));
    float *h_A, *h_B, *h_C, *h_C_prime;
    // Adding some syntaxic sugar to
    // explicity show that malloc returns a pointer
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    h_C_prime = (float *)malloc(nBytes);

    // Initialize the arrays on the host
    double iStart = seconds();
    initializeData(h_A, N);
    initializeData(h_B, N);
    double iElaps = seconds() - iStart;
    printf("Init time for vectors: %.5f sec\n", iElaps);

    // memset comes from the std C library
    memset(h_C, 0, nBytes);
    memset(h_C_prime, 0, nBytes);

    // Vector addition on the host
    double iStart2 = seconds();
    vectorAddOnHost(h_A, h_B, h_C, N);
    double iElaps2 = seconds() - iStart2;
    printf("Vector addition on the host: %.5f sec\n", iElaps2);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // Assigning the number of threads per block and number of blocks
    int numThreadsPerBlock = 32;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    double iStart3 = seconds();
    vectorAddOnDevice<<<numBlocks, numThreadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double iElaps3 = seconds() - iStart3;
    printf("Vector addition on the device: %.5f sec\n", iElaps3);

    // Copy result from device to host
    cudaMemcpy(h_C_prime, d_C, nBytes, cudaMemcpyDeviceToHost);
    // Check result
    checkResult(h_C, h_C_prime, N);
    // Free device global memory
    // Not need to do `cudaFree` on each individual pointer
    // as cudaDeviceReset will do it
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217
    cudaDeviceReset();
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_prime);

    // Reset device and exit
    printf("Done\n");
    return 0;
}