#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <string.h>

/*
 * This code implements the interleaved-paired approachs for
 * parallel reduction in CUDA with some unrolling of loops.
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

int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1)
        return data[0];
    // renew the stride
    int const stride = size / 2;
    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }
    // call recursively
    return recursiveReduce(data, stride);
}

// Interleaved Pair Implementation with warp and data block unrolling
__global__ void reduceInterleavedUnroll(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int block_start_idx = blockIdx.x * blockDim.x * 2;

    int *idata = g_idata + block_start_idx;

    // unroll 2 data blocks
    if (block_start_idx + tid + blockDim.x < n)
    {
        g_idata[block_start_idx + tid] += g_idata[block_start_idx + tid + blockDim.x];
    }
    __syncthreads();

    // In-place reduction in global memory using a decreasing stride.
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        // The first 'stride' threads are active.
        if (tid < stride)
        {
            // We need to map the active threads [0, stride-1] to the correct
            // interleaved indices from the previous step.
            // The 'multiplier' determines how far apart the elements are.
            // Ex: stride=256 -> multiplier=2. We add elements at (0,1), (2,3), ...
            // Ex: stride=128 -> multiplier=4. We add elements at (0,2), (4,6), ...
            unsigned int multiplier = (blockDim.x / stride);

            unsigned int write_local_idx = tid * multiplier;
            unsigned int read_local_idx = write_local_idx + (multiplier / 2);

            // We only need to check the read_idx, as the write_idx is guaranteed to be smaller.
            if (block_start_idx + read_local_idx < n)
            {
                idata[write_local_idx] += idata[read_local_idx];
            }
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        int my_sum = 0;
        if (blockDim.x > 32)
        {
            // If the loop ran, the 64 partial sums are interleaved in global memory.
            // We must read from two interleaved locations.
            unsigned int multiplier = blockDim.x / 32;
            unsigned int read_idx1 = tid * multiplier;
            unsigned int read_idx2 = read_idx1 + (multiplier / 2);

            // Check bounds before reading from global memory
            if (block_start_idx + read_idx1 < n)
                my_sum = idata[read_idx1];
            if (block_start_idx + read_idx2 < n)
                my_sum += idata[read_idx2];
        }
        else
        {
            // If the block size was 32 or less, the loop was skipped.
            // Data is at its original position in global memory.
            if (block_start_idx + tid < n)
                my_sum = idata[tid];
        }

        // This full sequence correctly sums the 32 values in `my_sum`.
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 16);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 8);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 4);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 2);
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, 1);

        // Thread 0 writes the final result.
        if (tid == 0)
        {
            g_odata[blockIdx.x] = my_sum;
        }
    }
}

void test_reduction()
{
    printf("\n--- Testing Parallel Reduction (Interleaved) ---\n");

    const int block_size = 256;

    std::vector<size_t> problem_sizes;
    problem_sizes.push_back(1);
    problem_sizes.push_back(block_size / 2);
    problem_sizes.push_back(block_size);
    problem_sizes.push_back(block_size + 1);
    problem_sizes.push_back(1024 * 1024);
    problem_sizes.push_back(1024 * 1024 + 13);
    problem_sizes.push_back(4 * 1024 * 1024);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (size_t N : problem_sizes)
    {
        if (N == 0)
        {
            printf("\nSkipping N = 0 (no elements to reduce)\n");
            continue;
        }

        unsigned int num_blocks = (N + block_size - 1) / block_size;
        dim3 gridDim(num_blocks);
        dim3 blockDim(block_size);

        size_t nBytes_input = N * sizeof(int);
        size_t nBytes_output_partial_sums = num_blocks * sizeof(int);

        printf("\nProblem Size N = %zu elements\n", N);
        printf("Configuration: %u blocks, %d threads/block\n", num_blocks, block_size);
        printf("Input data: %.2f MiB, Output (block sums): %.2f KiB\n",
               (double)nBytes_input / (1024.0 * 1024.0),
               (double)nBytes_output_partial_sums / 1024.0);

        std::vector<int> h_idata(N);
        int *d_idata = nullptr;
        int *d_odata = nullptr;

        CUDA_CHECK(cudaMalloc(&d_idata, nBytes_input));
        CUDA_CHECK(cudaMalloc(&d_odata, nBytes_output_partial_sums));
        CUDA_CHECK(cudaMemset(d_odata, 0, nBytes_output_partial_sums));

        long long expected_sum_cpu = 0;
        for (size_t i = 0; i < N; ++i)
        {
            h_idata[i] = (int)((i % 200) - 100);
            expected_sum_cpu += h_idata[i];
        }

        CUDA_CHECK(cudaMemcpy(d_idata, h_idata.data(), nBytes_input, cudaMemcpyHostToDevice));

        // Warmup run (good practice, especially for timing)
        reduceInterleavedUnroll<<<gridDim, blockDim>>>(d_idata, d_odata, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(d_idata, h_idata.data(), nBytes_input, cudaMemcpyHostToDevice));

        // Timed execution
        CUDA_CHECK(cudaEventRecord(start, 0));
        reduceInterleavedUnroll<<<gridDim, blockDim>>>(d_idata, d_odata, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        std::vector<int> h_odata_results(num_blocks);
        CUDA_CHECK(cudaMemcpy(h_odata_results.data(), d_odata, nBytes_output_partial_sums, cudaMemcpyDeviceToHost));

        long long gpu_final_sum = 0;
        for (unsigned int i = 0; i < num_blocks; ++i)
        {
            gpu_final_sum += h_odata_results[i];
        }

        // Verification
        printf("CPU Sum (expected): %lld\n", expected_sum_cpu);
        printf("GPU Sum (actual):   %lld\n", gpu_final_sum);
        if (expected_sum_cpu == gpu_final_sum)
        {
            printf("Verification: PASS\n");
        }
        else
        {
            printf("Verification: FAIL\n");
            fprintf(stderr, "Error: Mismatch for N = %zu. CPU sum: %lld, GPU sum: %lld\n", N, expected_sum_cpu, gpu_final_sum);
        }

        // Performance Calculation
        double seconds = milliseconds / 1000.0;
        double elements_per_sec = (double)N / seconds;
        double g_elements_per_sec = elements_per_sec / 1e9;

        // Effective Bandwidth Calculation:
        double total_bytes_processed = (double)nBytes_input + (double)nBytes_output_partial_sums;
        double effective_bandwidth_gibps = (total_bytes_processed / seconds) / (1024.0 * 1024.0 * 1024.0); // GiB/s

        printf("Elapsed Time: %.3f ms\n", milliseconds);
        printf("Performance:  %.3f GigaElements/s\n", g_elements_per_sec);
        printf("Effective Bandwidth (kernel part): %.3f GiB/s\n", effective_bandwidth_gibps);

        // Cleanup device memory for this problem size
        CUDA_CHECK(cudaFree(d_idata));
        CUDA_CHECK(cudaFree(d_odata));
    }

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    printf("-----------------------------------------------------\n");
}

int main()
{
    int deviceId;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    printf("Using CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Total Global Memory: %.2f MiB\n", prop.totalGlobalMem / (1024.0 * 1024.0));

    printf("Unrooling 2 data blocks in the kernel.\n");
    test_reduction();

    CUDA_CHECK(cudaDeviceReset());

    printf("\nDone.\n");
    return 0;
}