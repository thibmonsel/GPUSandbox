#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include <sys/time.h>
#include "../common/common.h"

/*
An example of airhtmetic and memory latency hiding.
*/

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

// Function to get basic device properties (Max Warps per SM)
int getMaxWarpsPerSM()
{
    cudaDeviceProp prop;
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warps per SM = Max Threads per SM / %d = %d\n",
           prop.warpSize, prop.maxThreadsPerMultiProcessor / prop.warpSize);
    return prop.maxThreadsPerMultiProcessor / prop.warpSize;
}

// Kernel performing dependent arithmetic operations
__global__ void arithmetic_latency_kernel(float *data, int num_iterations, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float val = data[idx];
        const float c1 = 1.000001f; // Slightly more than 1
        const float c2 = 0.000001f; // Small constant

        // Chain of dependent operations - latency cannot be hidden within a thread
        for (int i = 0; i < num_iterations; ++i)
        {
            val = val * c1 + c2; // FMA or Mul+Add - has latency
        }
        data[idx] = val;
    }
}

void test_arithmetic_latency(int max_warps_per_sm)
{
    printf("\n--- Testing Arithmetic Latency Hiding ---\n");

    const int block_size = 256; // Common block size
    const int iterations = 500; // Number of dependent ops per thread

    // We expect performance (GFLOPS) to increase as N grows,
    // until we have enough warps to hide latency.
    std::vector<size_t> problem_sizes;
    problem_sizes.push_back(1024 * 32);        // Small N
    problem_sizes.push_back(1024 * 256);       // Medium N
    problem_sizes.push_back(1024 * 1024);      // Large N
    problem_sizes.push_back(1024 * 1024 * 8);  // Very Large N
    problem_sizes.push_back(1024 * 1024 * 16); // Even Larger N

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (size_t N : problem_sizes)
    {
        dim3 blockDim(block_size);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
        size_t total_threads = (size_t)gridDim.x * blockDim.x;
        size_t nBytes = N * sizeof(float);

        printf("\nProblem Size N = %zu (Threads = %zu, Blocks = %u)\n", N, total_threads, gridDim.x);

        float *d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, nBytes));

        std::vector<float> h_data(N, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), nBytes, cudaMemcpyHostToDevice));

        // Warmup run (optional but good practice)
        arithmetic_latency_kernel<<<gridDim, blockDim>>>(d_data, iterations, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed run
        CUDA_CHECK(cudaEventRecord(start, 0));
        arithmetic_latency_kernel<<<gridDim, blockDim>>>(d_data, iterations, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Calculate performance
        // Each thread does 'iterations' multiply-adds (2 FLOPS each)
        double total_flops = (double)N * iterations * 2.0;
        double seconds = milliseconds / 1000.0;
        double gflops = (total_flops / seconds) / 1e9;

        printf("Elapsed Time: %.3f ms\n", milliseconds);
        printf("Performance:  %.3f GFLOPS\n", gflops);

        // Cleanup for this size
        CUDA_CHECK(cudaFree(d_data));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    printf("-------------------------------------------\n");
}

// Kernel performing simple memory copy - dominated by memory latency
__global__ void memory_latency_kernel(const float *input, float *output, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        // Read from global memory (high latency operation)
        float val = input[idx];
        val = val + 1.0f;
        // Write to global memory (another high latency operation)
        output[idx] = val;
    }
}

void test_memory_latency(int max_warps_per_sm)
{
    printf("\n--- Testing Memory Latency Hiding ---\n");

    const int block_size = 256;

    // We expect performance (Bandwidth GB/s) to increase as N grows,
    // requiring many more warps than the arithmetic case.
    std::vector<size_t> problem_sizes;
    // Start larger, as memory bandwidth needs more data
    problem_sizes.push_back(1024 * 1024 * 4);   // Medium N
    problem_sizes.push_back(1024 * 1024 * 16);  // Large N
    problem_sizes.push_back(1024 * 1024 * 64);  // Very Large N
    problem_sizes.push_back(1024 * 1024 * 128); // Even Larger N
    problem_sizes.push_back(1024 * 1024 * 256); // Massive N

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (size_t N : problem_sizes)
    {
        // Ensure N is large enough to likely stress memory subsystem
        if (N < 1024 * 1024)
        {
            printf("\nSkipping small N = %zu for memory test\n", N);
            continue;
        }

        dim3 blockDim(block_size);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
        size_t total_threads = (size_t)gridDim.x * blockDim.x;
        size_t nBytes = N * sizeof(float);

        printf("\nProblem Size N = %zu (Threads = %zu, Blocks = %u, %.2f MB)\n",
               N, total_threads, gridDim.x, (double)nBytes / (1024.0 * 1024.0));

        float *d_input = nullptr, *d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, nBytes));
        CUDA_CHECK(cudaMalloc(&d_output, nBytes));

        std::vector<float> h_input(N);
        for (size_t i = 0; i < N; ++i)
            h_input[i] = (float)i;
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), nBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_output, 0, nBytes)); // Clear output

        // Warmup run
        memory_latency_kernel<<<gridDim, blockDim>>>(d_input, d_output, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start, 0));
        memory_latency_kernel<<<gridDim, blockDim>>>(d_input, d_output, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Calculate performance (Effective Bandwidth)
        // We read N floats and write N floats. Total = 2 * N * sizeof(float) bytes
        double total_bytes = (double)2.0 * N * sizeof(float);
        double seconds = milliseconds / 1000.0;
        double gb_per_sec = (total_bytes / seconds) / (1024.0 * 1024.0 * 1024.0); // GiB/s

        printf("Elapsed Time: %.3f ms\n", milliseconds);
        printf("Effective Bandwidth: %.3f GiB/s\n", gb_per_sec);

        // Cleanup for this size
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    printf("-----------------------------------------\n");
}

int main()
{
    int max_warps = getMaxWarpsPerSM();

    test_arithmetic_latency(max_warps);
    // test_memory_latency(max_warps);

    CUDA_CHECK(cudaDeviceReset());

    printf("\nDone.\n");
    return 0;
}