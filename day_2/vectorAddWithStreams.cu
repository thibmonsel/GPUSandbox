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

    // This is no limit to the number of streams
    // but the hardware can only support a number a
    // certain number of concurrent kernel.
    int numStreams = 16;

    // Setting size of my arrays N = 1M elements
    const int N = 1 << 20;
    size_t vectorSizeBytes = N * sizeof(float);
    // number of threads per block
    int threadsPerBlock = 256;

    // Calculate base chunk size (integer division)
    // Handle cases where N might be smaller than numStreams
    int baseChunkSize = (N > 0) ? (N + numStreams - 1) / numStreams : 0;
    printf("Vector size: %d that will be of size %d bytes (sizeof(float)=%d) \n", N, vectorSizeBytes, sizeof(float));
    printf("Base chunk size (approx): %d elements\n", baseChunkSize);

    float *h_A, *h_B, *h_C, *h_C_prime;
    cudaMallocHost((void **)&h_A, vectorSizeBytes);
    cudaMallocHost((void **)&h_B, vectorSizeBytes);
    cudaMallocHost((void **)&h_C, vectorSizeBytes);
    cudaMallocHost((void **)&h_C_prime, vectorSizeBytes);

    // Initialize the arrays on the host
    double iStart = seconds();
    initializeData(h_A, N);
    initializeData(h_B, N);
    double iElaps = seconds() - iStart;
    printf("Init time for vectors: %.5f sec\n", iElaps);

    // memset comes from the std C library
    memset(h_C, 0, vectorSizeBytes);
    memset(h_C_prime, 0, vectorSizeBytes);

    // Vector addition on the host
    double iStart2 = seconds();
    vectorAddOnHost(h_A, h_B, h_C, N);
    double iElaps2 = seconds() - iStart2;
    printf("Vector addition on the host: %.5f sec\n", iElaps2);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, vectorSizeBytes);
    cudaMalloc((float **)&d_B, vectorSizeBytes);
    cudaMalloc((float **)&d_C, vectorSizeBytes);

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(numStreams * sizeof(
                                                                    cudaStream_t));

    for (int i = 0; i < numStreams; i++)
    {
        cudaStreamCreate(&(streams[i]));
    }

    // --- Timing ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Execute with Streams ---
    printf("Starting GPU vector add with streams...\n");
    cudaEventRecord(start); // Record start time

    int currentOffset = 0; // Keep track of the starting position for the next chunk
    for (int i = 0; i < numStreams; ++i)
    {
        // Calculate the number of elements for THIS stream's chunk
        // Make sure we don't go past the end of the vector N
        int elementsInThisChunk = std::min(baseChunkSize, N - currentOffset);

        // If elementsInThisChunk is 0 or negative, it means we've processed all elements
        // in previous streams (this can happen if n < numStreams). Break the loop.
        if (elementsInThisChunk <= 0)
        {
            printf("Stream %d has no work (all elements assigned).\n", i);
            break;
        }

        size_t currentChunkSizeBytes = elementsInThisChunk * sizeof(float);
        printf("Stream %d: offset=%d, elements=%d, bytes=%.2f KB\n",
               i, currentOffset, elementsInThisChunk, (float)currentChunkSizeBytes / 1024.0f);

        // 1. Asynchronous Host-to-Device Copy for chunk i
        cudaMemcpyAsync(d_A + currentOffset, h_A + currentOffset, currentChunkSizeBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B + currentOffset, h_B + currentOffset, currentChunkSizeBytes, cudaMemcpyHostToDevice, streams[i]);

        // 2. Launch Kernel for chunk i in its stream
        int blocksPerGrid = (elementsInThisChunk + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksPerGrid > 0)
        { // Only launch kernel if there's work to do
            vectorAddOnDevice<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
                d_A + currentOffset,
                d_A + currentOffset,
                d_C + currentOffset,
                elementsInThisChunk // Pass the actual size for this chunk
            );
            // Check for kernel launch errors (asynchronous)
            cudaError_t kernelError = cudaGetLastError();
            if (kernelError != cudaSuccess)
            {
                fprintf(stderr, "Kernel launch error in stream %d: %s\n", i, cudaGetErrorString(kernelError));
            }
        }
        else
        {
            printf("Stream %d: Skipping kernel launch (0 blocks).\n", i);
        }

        // 3. Asynchronous Device-to-Host Copy for chunk i
        cudaMemcpyAsync(h_C_prime + currentOffset, d_C + currentOffset, currentChunkSizeBytes, cudaMemcpyDeviceToHost, streams[i]);

        // Update offset for the next stream's chunk
        currentOffset += elementsInThisChunk;
    }
    // Ensure we have processed all elements if the loop finished normally
    if (currentOffset != N && N > 0)
    {
        fprintf(stderr, "Error: Loop finished but not all elements processed. currentOffset=%d, n=%d\n", currentOffset, N);
        // Handle error appropriately - maybe exit or throw exception
    }

    // --- Synchronization ---
    cudaDeviceSynchronize();

    // --- Timing Stop ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU execution time (streams): %.3f ms\n", milliseconds);
    if (milliseconds > 0)
    {
        double totalBytesTransferred = 3.0 * vectorSizeBytes; // 2 reads H->D, 1 write D->H
        printf("Effective Bandwidth: %.2f GB/s\n", (totalBytesTransferred / (milliseconds / 1000.0)) / (1024.0 * 1024.0 * 1024.0));
    }

    checkResult(h_C, h_C_prime, N);

    cudaDeviceReset();

    // Reset device and exit
    printf("Done\n");
    return 0;
}