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

// --- Optimized Kernel: Grid-Stride Loop + float4 ---
__global__ void vectorAddOptimized(const float4 *__restrict__ A,
                                   const float4 *__restrict__ B,
                                   float4 *__restrict__ C,
                                   const int N_chunk_float4)
{
    // // Using __restrict__ as pointers don't alias
    // Global thread ID calculating items of size float4
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Grid stride in units of float4 items
    const int stride = gridDim.x * blockDim.x;

    // Use grid-stride loop over float4 elements
    // Now each thread is now responsible for processing multiple
    // elements distributed throughout the array/chunk, not just one.
    for (int i = tid; i < N_chunk_float4; i += stride)
    {
        // Perform 4 additions per loop iteration per thread
        // Loading 4 floats at once by using the type float4
        // Then with its attribute .x, .y, .z, .w we can access each float
        // This an optimized type in CUDA that hold 4 floats and is
        // vectorized memory access (float4 uses one memory instruction to load 4 floats)
        const float4 a_vals = A[i];
        const float4 b_vals = B[i];
        float4 c_vals;
        c_vals.x = a_vals.x + b_vals.x;
        c_vals.y = a_vals.y + b_vals.y;
        c_vals.z = a_vals.z + b_vals.z;
        c_vals.w = a_vals.w + b_vals.w;
        C[i] = c_vals;
    }
}

int main(void)
{
    printf("Starting Vector Add with CUDA Streams (Optimized Kernel)...\n");

    // --- Configuration ---
    int numStreams = 4;
    const int N_requested = 1 << 20;
    int threadsPerBlock = 256;

    // Ensure N is divisible by 4 for float4 alignment
    const int N = (N_requested / 4) * 4;
    if (N != N_requested)
    {
        printf("Adjusted N from %d to %d for float4 alignment.\n", N_requested, N);
    }

    size_t vectorSizeBytes = (size_t)N * sizeof(float);
    printf("Configuration:\n");
    printf("Vector elements (N): %d (%.2f Million)\n", N, (float)N / (1 << 20));
    printf("Vector size: %zu bytes (%.2f MB)\n", vectorSizeBytes, (float)vectorSizeBytes / (1 << 20));
    printf("CUDA Streams: %d\n", numStreams);
    printf("Threads per Block: %d\n", threadsPerBlock);
    printf("Using float4 vectorized kernel.\n");

    // --- Host Memory Allocation (Pinned Memory Recommended for Async Transfers) ---
    float *h_A, *h_B, *h_C, *h_C_prime;
    printf("Allocating Pinned Host Memory...\n");
    CUDA_CHECK(cudaMallocHost((void **)&h_A, vectorSizeBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_B, vectorSizeBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_C, vectorSizeBytes));       // Host reference result
    CUDA_CHECK(cudaMallocHost((void **)&h_C_prime, vectorSizeBytes)); // GPU result copied back
    if (!h_A || !h_B || !h_C || !h_C_prime)
    { /* Error handling */
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
    // Allocate as floats, but treat as float4 in kernel/pointers
    float *d_A_base, *d_B_base, *d_C_base;
    printf("Allocating Device Memory...\n");
    CUDA_CHECK(cudaMalloc((void **)&d_A_base, vectorSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B_base, vectorSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C_base, vectorSizeBytes));

    // --- Create CUDA Streams ---
    printf("Creating %d CUDA streams...\n", numStreams);
    cudaStream_t *streams = (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));
    if (!streams)
    { /* Error handling */
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
    printf("\nStarting GPU execution with %d streams (Optimized Kernel)...\n", numStreams);

    CUDA_CHECK(cudaEventRecord(start_event, 0)); // Record start time in default stream

    int currentOffset = 0; // Track offset in float elements
    for (int i = 0; i < numStreams; ++i)
    {
        // Calculate elements (floats) for this chunk, ensuring divisibility by 4
        int elementsInThisChunk = N / numStreams;
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

        // Adjust chunk size to be multiple of 4 for float4 processing
        // This might make chunks slightly uneven, but simplifies kernel logic
        int remainder4 = elementsInThisChunk % 4;
        if (remainder4 != 0)
        {
            // Add padding to make it divisible by 4.
            // Ensure we don't go past N!
            int needed = 4 - remainder4;
            if (currentOffset + elementsInThisChunk + needed <= N)
            {
                elementsInThisChunk += needed;
            }
            else
            {
                // If adding padding would exceed N, just take what's left up to the next multiple of 4
                elementsInThisChunk -= remainder4;
            }
            // Re-check if last stream needs trimming if total exceeds N
            if (currentOffset + elementsInThisChunk > N)
            {
                elementsInThisChunk = N - currentOffset;
            }
        }

        // Ensure no processing if chunk is empty or negative (edge cases)
        if (elementsInThisChunk <= 0)
        {
            if (currentOffset < N)
            { // Only print warning if we haven't processed all N yet
                printf("Stream %d: Calculated zero or negative elements (%d), skipping.\n", i, elementsInThisChunk);
            }
            continue;
        }
        if (currentOffset >= N)
            continue; // Already processed everything

        int elementsInThisChunk_float4 = elementsInThisChunk / 4;
        size_t currentChunkSizeBytes = (size_t)elementsInThisChunk * sizeof(float);

        printf("Stream %d: Offset(floats)=%d, Elements(floats)=%d (float4=%d), Size=%.2f KB\n",
               i, currentOffset, elementsInThisChunk, elementsInThisChunk_float4, (float)currentChunkSizeBytes / 1024.0f);

        // Define pointers for the current chunk's start (base is float*)
        float *h_A_chunk = h_A + currentOffset;
        float *h_B_chunk = h_B + currentOffset;
        float *d_A_chunk_base = d_A_base + currentOffset; // Offset by float
        float *d_B_chunk_base = d_B_base + currentOffset;
        float *d_C_chunk_base = d_C_base + currentOffset;
        float *h_C_prime_chunk = h_C_prime + currentOffset;

        // Cast device pointers to float4* for the kernel
        float4 *d_A_chunk_f4 = (float4 *)d_A_chunk_base;
        float4 *d_B_chunk_f4 = (float4 *)d_B_chunk_base;
        float4 *d_C_chunk_f4 = (float4 *)d_C_chunk_base;

        // 1. Async H2D Copy (operates on bytes, original pointers are fine)
        CUDA_CHECK(cudaMemcpyAsync(d_A_chunk_base, h_A_chunk, currentChunkSizeBytes, cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_B_chunk_base, h_B_chunk, currentChunkSizeBytes, cudaMemcpyHostToDevice, streams[i]));

        // 2. Launch Kernel (Use float4 pointers and count)
        // Calculate blocks based on float4 elements
        int blocksPerGrid = (elementsInThisChunk_float4 + threadsPerBlock - 1) / threadsPerBlock;

        if (blocksPerGrid > 0 && elementsInThisChunk_float4 > 0)
        {
            vectorAddOptimized<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
                d_A_chunk_f4,
                d_B_chunk_f4,
                d_C_chunk_f4,
                elementsInThisChunk_float4); // Pass float4 count
            CUDA_CHECK(cudaGetLastError());  // Check launch error immediately
        }
        else
        {
            printf("Stream %d: Skipping kernel launch (0 blocks or 0 float4 elements).\n", i);
        }

        // 3. Async D2H Copy (operates on bytes)
        CUDA_CHECK(cudaMemcpyAsync(h_C_prime_chunk, d_C_chunk_base, currentChunkSizeBytes, cudaMemcpyDeviceToHost, streams[i]));

        // Update float offset for the next stream's chunk
        currentOffset += elementsInThisChunk;
    }

    // Sanity check after loop
    if (currentOffset != N)
    {
        fprintf(stderr, "Error: Loop finished but final float offset (%d) does not equal N (%d)!\n", currentOffset, N);
        // Cleanup...
        return EXIT_FAILURE;
    }

    // --- Synchronization ---
    printf("Synchronizing device...\n");
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all streams to finish
    printf("Device synchronization complete.\n");

    // --- Timing Stop & Calculation ---
    CUDA_CHECK(cudaEventRecord(stop_event, 0));   // Record stop time
    CUDA_CHECK(cudaEventSynchronize(stop_event)); // Wait for stop event to finish
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    printf("\n--- GPU Performance ---\n");
    printf("Total GPU execution time (H2D Copy + Kernel + D2H Copy): %.3f ms\n", milliseconds);

    if (milliseconds > 0)
    {
        double totalBytesTransferred = 3.0 * vectorSizeBytes; // A, B H->D, C D->H
        double secondsElapsed = milliseconds / 1000.0;
        double gibPerSecond = (totalBytesTransferred / secondsElapsed) / (1024.0 * 1024.0 * 1024.0);
        printf("Effective Memory Bandwidth (Combined H2D + D2H): %.3f GiB/s\n", gibPerSecond);

        // Estimate kernel GFLOPS (N additions = N FLOPs)
        double gflops = (double)N / secondsElapsed / 1e9;
        printf("Kernel Estimated Performance: %.3f GFLOPS\n", gflops);
    }

    // --- Verification ---
    printf("\n--- Verification ---\n");
    checkResult(h_C, h_C_prime, N); // Verify using original float pointers

    // --- Cleanup ---
    printf("\nCleaning up resources...\n");
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    for (int i = 0; i < numStreams; i++)
    {
        if (streams[i])
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    if (d_A_base)
        CUDA_CHECK(cudaFree(d_A_base));
    if (d_B_base)
        CUDA_CHECK(cudaFree(d_B_base));
    if (d_C_base)
        CUDA_CHECK(cudaFree(d_C_base));
    if (h_A)
        CUDA_CHECK(cudaFreeHost(h_A));
    if (h_B)
        CUDA_CHECK(cudaFreeHost(h_B));
    if (h_C)
        CUDA_CHECK(cudaFreeHost(h_C));
    if (h_C_prime)
        CUDA_CHECK(cudaFreeHost(h_C_prime));

    printf("Done\n");
    return 0;
}