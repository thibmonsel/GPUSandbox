#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../common/common.h"

#define CUDA_CHECK(call)                                                                                                 \
    do                                                                                                                   \
    {                                                                                                                    \
        cudaError_t err = call;                                                                                          \
        if (err != cudaSuccess)                                                                                          \
        {                                                                                                                \
            fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
            /* Attempt to reset device before exiting might help cleanup */                                              \
            cudaDeviceReset();                                                                                           \
            exit(EXIT_FAILURE);                                                                                          \
        }                                                                                                                \
    } while (0)

void initializeData(float *ip, int size)
{
    srand((unsigned)time(NULL));
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
    bool match = true;
    int mismatch_count = 0;
    const int max_mismatches_to_print = 10;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            mismatch_count++;
            if (mismatch_count <= max_mismatches_to_print)
            {
                printf("Mismatch at index %d: host=%.5f gpu=%.5f diff=%.5e\n",
                       i, hostRef[i], gpuRef[i], abs(hostRef[i] - gpuRef[i]));
            }
        }
    }

    if (match)
    {
        printf("Arrays match.\n");
    }
    else
    {
        printf("ARRAYS DO NOT MATCH! Total mismatches: %d\n", mismatch_count);
    }
}

__global__ void vectorAddOnDevice(float *A, float *B, float *C, const int N_chunk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Important: Boundary check uses N_chunk, not the total N
    if (i < N_chunk)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    printf("Starting Vector Add with CUDA Streams...\n");

    // --- Configuration ---
    int numStreams = 2;
    const int N = 1 << 20;
    int threadsPerBlock = 256;

    size_t vectorSizeBytes = (size_t)N * sizeof(float);
    printf("Configuration:\n");
    printf("Vector elements (N): %d (%.2f Million)\n", N, (float)N / (1 << 20));
    printf("Vector size: %zu bytes (%.2f MB)\n", vectorSizeBytes, (float)vectorSizeBytes / (1 << 20));
    printf("CUDA Streams: %d\n", numStreams);
    printf("Threads per Block: %d\n", threadsPerBlock);

    // --- Host Memory Allocation (Pinned Memory Recommended for Async Transfers) ---
    float *h_A, *h_B, *h_C, *h_C_prime;
    printf("Allocating Pinned Host Memory...\n");
    CUDA_CHECK(cudaMallocHost((void **)&h_A, vectorSizeBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_B, vectorSizeBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_C, vectorSizeBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_C_prime, vectorSizeBytes));
    if (!h_A || !h_B || !h_C || !h_C_prime)
    {
        fprintf(stderr, "Failed to allocate host memory!\n");
        cudaDeviceReset();
        return EXIT_FAILURE;
    }

    // --- Initialize Host Data ---
    printf("Initializing host data...\n");
    double iStartInit = seconds();
    initializeData(h_A, N);
    initializeData(h_B, N);
    memset(h_C_prime, 0, vectorSizeBytes);
    double iEndInit = seconds();
    printf("Host data initialization time: %.5f sec\n", iEndInit - iStartInit);

    // --- Host Calculation (for verification) ---
    printf("Performing vector addition on host...\n");
    double iStartHost = seconds();
    vectorAddOnHost(h_A, h_B, h_C, N);
    double iEndHost = seconds();
    printf("Host vector addition time: %.5f sec\n", iEndHost - iStartHost);

    // --- Device Memory Allocation ---
    float *d_A, *d_B, *d_C;
    printf("Allocating Device Memory...\n");
    CUDA_CHECK(cudaMalloc((float **)&d_A, vectorSizeBytes));
    CUDA_CHECK(cudaMalloc((float **)&d_B, vectorSizeBytes));
    CUDA_CHECK(cudaMalloc((float **)&d_C, vectorSizeBytes));

    // --- Create CUDA Streams ---
    printf("Creating %d CUDA streams...\n", numStreams);
    cudaStream_t *streams = (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));
    if (!streams)
    {
        fprintf(stderr, "Failed to allocate memory for stream handles!\n");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        cudaFreeHost(h_C);
        cudaFreeHost(h_C_prime);
        cudaDeviceReset();
        return EXIT_FAILURE;
    }
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&(streams[i])));
    }

    // --- Create CUDA Events for Timing ---
    cudaEvent_t start_event, stop_event;
    printf("Creating CUDA events for timing...\n");
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // --- Execute Operations with Streams ---
    printf("\nStarting GPU execution with %d streams...\n", numStreams);

    // <<< Record Start Event >>>
    // Record in the default stream (0) before queueing any work.
    CUDA_CHECK(cudaEventRecord(start_event, 0));

    int currentOffset = 0;
    for (int i = 0; i < numStreams; ++i)
    {
        // Calculate the number of elements and size for THIS stream's chunk
        int elementsInThisChunk = N / numStreams; // Integer division gives base size
        int remainder = N % numStreams;
        if (i < remainder)
        { // Distribute remainder among the first few streams
            elementsInThisChunk++;
        }

        if (elementsInThisChunk <= 0)
        {
            // This shouldn't happen with the improved distribution, but good practice
            printf("Stream %d: No elements to process, skipping.\n", i);
            continue; // Skip to the next stream
        }

        size_t currentChunkSizeBytes = (size_t)elementsInThisChunk * sizeof(float);
        printf("  Stream %d: Offset=%d, Elements=%d, Size=%.2f KB\n",
               i, currentOffset, elementsInThisChunk, (float)currentChunkSizeBytes / 1024.0f);

        // Define pointers for the current chunk's start
        float *h_A_chunk = h_A + currentOffset;
        float *h_B_chunk = h_B + currentOffset;
        float *d_A_chunk = d_A + currentOffset;
        float *d_B_chunk = d_B + currentOffset;
        float *d_C_chunk = d_C + currentOffset;
        float *h_C_prime_chunk = h_C_prime + currentOffset;

        // 1. Asynchronous Host-to-Device Copy for chunk i
        CUDA_CHECK(cudaMemcpyAsync(d_A_chunk, h_A_chunk, currentChunkSizeBytes, cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_B_chunk, h_B_chunk, currentChunkSizeBytes, cudaMemcpyHostToDevice, streams[i]));

        // 2. Launch Kernel for chunk i in its stream
        int blocksPerGrid = (elementsInThisChunk + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksPerGrid > 0)
        {
            vectorAddOnDevice<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
                d_A_chunk,
                d_B_chunk,
                d_C_chunk,
                elementsInThisChunk);
            // Check for kernel launch errors (asynchronous) immediately
            CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            printf("  Stream %d: Skipping kernel launch (0 blocks).\n", i);
        }

        // 3. Asynchronous Device-to-Host Copy for chunk i's result
        CUDA_CHECK(cudaMemcpyAsync(h_C_prime_chunk, d_C_chunk, currentChunkSizeBytes, cudaMemcpyDeviceToHost, streams[i]));

        // Update offset for the next stream's chunk
        currentOffset += elementsInThisChunk;
    }

    // Sanity check: Ensure all elements were processed
    if (currentOffset != N)
    {
        fprintf(stderr, "Error: Loop finished but offset (%d) does not equal N (%d)!\n", currentOffset, N);
        // Handle error - cleanup and exit might be best
    }

    // --- Synchronization ---
    // Wait for all commands in all streams to complete before stopping the timer.
    printf("Synchronizing device...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Device synchronization complete.\n");

    // --- Timing Stop & Calculation ---
    // <<< Record Stop Event >>>
    // Record in the default stream after all work is done.
    CUDA_CHECK(cudaEventRecord(stop_event, 0));

    // Wait for the stop event itself to be processed before calculating time.
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("\n--- GPU Performance ---\n");
    printf("Total GPU execution time (H2D Copy + Kernel + D2H Copy): %.3f ms\n", milliseconds);

    if (milliseconds > 0)
    {
        // Calculate effective bandwidth: considers H->D for A & B, and D->H for C
        double totalBytesTransferred = 3.0 * vectorSizeBytes;
        double secondsElapsed = milliseconds / 1000.0;
        double gibPerSecond = (totalBytesTransferred / secondsElapsed) / (1024.0 * 1024.0 * 1024.0);
        printf("Effective Memory Bandwidth (Combined H2D + D2H): %.3f GiB/s\n", gibPerSecond);

        // Estimate kernel GFLOPS (N additions = N FLOPs)
        double gflops = (double)N / secondsElapsed / 1e9;
        printf("Kernel Estimated Performance: %.3f GFLOPS\n", gflops);
    }

    // --- Verification ---
    printf("\n--- Verification ---\n");
    checkResult(h_C, h_C_prime, N);

    // --- Cleanup ---
    printf("\nCleaning up resources...\n");
    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    // Destroy streams
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams); // Free the array holding stream handles

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free pinned host memory
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFreeHost(h_C_prime));

    printf("Done\n");
    return 0;
}