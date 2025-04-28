#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../common/common.h"

/*
An example of warp divergence. These kernels will show the branch efficiency.
*/

// Adjust this value if needed to make timing differences clearer
#define WORKLOAD_ITERATIONS 500

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

// Kernel 1: Causes divergence within each warp (with more work)
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f;
    float b = 0.0f;

    // More substantial work inside branches
    if (tid % 2 == 0)
    {
        // Path A work
        a = 100.0f;
        for (int i = 0; i < WORKLOAD_ITERATIONS; ++i)
        {
            a = (a / 1.01f) + (float)i - (float)(tid & 1); // Example computation
        }
    }
    else
    {
        // Path B work (similar amount of computation)
        b = 200.0f;
        for (int i = 0; i < WORKLOAD_ITERATIONS; ++i)
        {
            b = (b / 1.02f) + (float)i + (float)(tid & 1); // Example computation
        }
    }
    // Divergent warps execute Path A then Path B serially.
    c[tid] = a + b;
}

// Kernel 2: No intra-warp divergence (with more work)
__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f;
    float b = 0.0f;

    // Condition based on warp ID
    if ((tid / warpSize) % 2 == 0)
    {
        // Path A work (Same as Kernel 1's Path A)
        a = 100.0f;
        for (int i = 0; i < WORKLOAD_ITERATIONS; ++i)
        {
            a = (a / 1.01f) + (float)i - (float)(tid & 1);
        }
    }
    else
    {
        // Path B work (Same as Kernel 1's Path B)
        b = 200.0f;
        for (int i = 0; i < WORKLOAD_ITERATIONS; ++i)
        {
            b = (b / 1.02f) + (float)i + (float)(tid & 1);
        }
    }
    // Each warp executes ONLY Path A OR Path B. No serialization within the warp.
    c[tid] = a + b;
}

// Kernel 3: Separate Ifs, high divergence (with more work)
__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia = 0.0f;
    float ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    // More substantial work inside branches
    if (ipred)
    {
        // Path A' work (Same as Kernel 1's Path A)
        ia = 100.0f;
        for (int i = 0; i < WORKLOAD_ITERATIONS; ++i)
        {
            ia = (ia / 1.01f) + (float)i - (float)(tid & 1);
        }
    }

    // Separate check - still causes serialization if ipred differs within warp
    if (!ipred)
    {
        // Path B' work (Same as Kernel 1's Path B)
        ib = 200.0f;
        for (int i = 0; i < WORKLOAD_ITERATIONS; ++i)
        {
            ib = (ib / 1.02f) + (float)i + (float)(tid & 1);
        }
    }
    // Divergent warps execute Path A' then Path B' serially.
    c[tid] = ia + ib;
}

int main(void)
{

    // --- Configuration ---
    // Increased nx slightly more just in case, but workload increase is main change
    const int nx = 32 * 1024 * 1024;
    size_t nBytes = (size_t)nx * sizeof(float);
    printf("Array size: %d elements (%zu bytes)\n", nx, nBytes);
    printf("Workload iterations per branch: %d\n", WORKLOAD_ITERATIONS);

    int dimx = 256;
    dim3 block(dimx, 1, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1, 1);
    printf("Grid: (%u, %u, %u), Block: (%u, %u, %u)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // --- Device Memory Allocation ---
    float *d_C;
    printf("Allocating Device Memory (%zu bytes)...\n", nBytes);
    CUDA_CHECK(cudaMalloc((float **)&d_C, nBytes));
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));

    // --- Timing Setup ---
    cudaEvent_t start, stop;
    float milliseconds = 0;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Warmup Kernel ---
    // Use kernel 2 for warmup as it *should* be faster overall
    printf("\nRunning warmup kernel (mathKernel2)...\n");
    CUDA_CHECK(cudaEventRecord(start, 0));
    mathKernel2<<<grid, block>>>(d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Warmup <<< %4u, %4u >>> elapsed %.3f ms\n", grid.x, block.x, milliseconds);

    // --- Run Kernel 1 (High Divergence) ---
    printf("\nRunning mathKernel1 (High Intra-Warp Divergence)...\n");
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure GPU is idle before starting timer
    CUDA_CHECK(cudaEventRecord(start, 0));
    mathKernel1<<<grid, block>>>(d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for kernel 1 to finish
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("mathKernel1 <<< %4u, %4u >>> elapsed %.3f ms (Expect higher time)\n", grid.x, block.x, milliseconds);

    // --- Run Kernel 2 (Inter-Warp Divergence Only) ---
    // *** This is the crucial measurement ***
    printf("\nRunning mathKernel2 (Inter-Warp Divergence Only)...\n");
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure GPU is idle before starting timer
    CUDA_CHECK(cudaEventRecord(start, 0));
    mathKernel2<<<grid, block>>>(d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for kernel 2 to finish
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    // *** This time should now be significantly lower than Kernel 1 and 3 ***
    printf("mathKernel2 <<< %4u, %4u >>> elapsed %.3f ms (Expect lower time)\n", grid.x, block.x, milliseconds);

    // --- Run Kernel 3 (Separate Ifs, High Divergence) ---
    printf("\nRunning mathKernel3 (Separate Ifs, High Intra-Warp Divergence)...\n");
    CUDA_CHECK(cudaMemset(d_C, 0, nBytes));
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure GPU is idle before starting timer
    CUDA_CHECK(cudaEventRecord(start, 0));
    mathKernel3<<<grid, block>>>(d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for kernel 3 to finish
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("mathKernel3 <<< %4u, %4u >>> elapsed %.3f ms (Expect time similar to Kernel 1)\n", grid.x, block.x, milliseconds);

    // --- Cleanup ---
    printf("\nCleaning up resources...\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_C));

    // cudaDeviceReset(); // Optional

    printf("Done\n");
    return 0;
}