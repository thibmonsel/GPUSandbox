#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../common/common.h"

/*
An example of a matrix addition on the host and device. Using a simple kernel to add two
matrices together. Typically, a matrix is stored linearly in global memory with a row-major
approach. With this representation, we can use with ease the 1D grid and 1D block although
higher 2D grid/blocks can be used.
*/

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
    // generate different seed for random number
    srand((unsigned)time(NULL));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
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
        printf("Arrays match.\n\n");
}

void matrixAddOnHost(float *A, float *B, float *C, const int nRow, const int nCol)
{
    // View matrix as row master view Row 0 - Row 1 - .... - Row 6 and Row i : shape j
    for (int ix = 0; ix < nCol; ix++)
    {
        for (int iy = 0; iy < nRow; iy++)
        {
            C[iy * nRow + ix] = A[iy * nRow + ix] + B[iy * nRow + ix];
        }
    }
}

__global__ void matrixAddonDevice(float *A, float *B, float *C, const int nRow, const int nCol)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nRow * nCol)
    {
        int ix = tid % nCol;
        int iy = tid / nRow;
        C[iy * nCol + ix] = A[iy * nCol + ix] + B[iy * nCol + ix];
    }
}
int main(void)
{

    // --- Configuration ---
    const int nRow = 1 << 14;
    const int nCol = 1 << 14;
    const int N = nRow * nCol;

    size_t nBytes = (size_t)N * sizeof(float);
    printf("Matrix size: nRow %d nCol %d\n", nRow, nCol);

    int numThreadsPerBlock = 256;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // --- Host Memory Allocation ---
    float *h_A, *h_B, *h_C, *h_C_prime;
    printf("Allocating host memory (%zu bytes)...\n", 4 * nBytes);
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    h_C_prime = (float *)malloc(nBytes);
    if (!h_A || !h_B || !h_C || !h_C_prime)
    {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return EXIT_FAILURE;
    }

    // --- Initialize Host Data ---
    printf("Initializing host data...\n");
    double iStartInit = seconds();
    initializeData(h_A, N);
    initializeData(h_B, N);
    memset(h_C_prime, 0, nBytes);
    double iEndInit = seconds();
    printf("Host data initialization time: %.5f sec\n", iEndInit - iStartInit);

    // --- Host Calculation (for verification) ---
    printf("Performing matrix addition on host...\n");
    double iStartHost = seconds();
    matrixAddOnHost(h_A, h_B, h_C, nRow, nCol);
    double iEndHost = seconds();
    printf("Host vector addition time: %.5f sec\n", iEndHost - iStartHost);

    // --- Device Memory Allocation ---
    float *d_A, *d_B, *d_C;
    printf("Allocating Device Memory (%zu bytes)...\n", nBytes);
    CUDA_CHECK(cudaMalloc((float **)&d_A, nBytes));
    CUDA_CHECK(cudaMalloc((float **)&d_B, nBytes));
    CUDA_CHECK(cudaMalloc((float **)&d_C, nBytes));

    // --- GPU Work & Timing (H2D + Kernel + D2H) ---
    cudaEvent_t start_gpu_total, stop_gpu_total;
    printf("\nCreating CUDA events for overall GPU timing...\n");
    CUDA_CHECK(cudaEventCreate(&start_gpu_total));
    CUDA_CHECK(cudaEventCreate(&stop_gpu_total));

    printf("Starting GPU operations and recording events...\n");

    // <<< Record Start Event >>> (Before *first* GPU operation)
    CUDA_CHECK(cudaEventRecord(start_gpu_total, 0)); // 0 for default stream

    // 1. H2D Data Transfer
    printf("  Copying data from host to device (A)...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    printf("  Copying data from host to device (B)...\n");
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // 2. GPU Kernel Execution
    printf("  Launching kernel...\n");
    matrixAddonDevice<<<numBlocks, numThreadsPerBlock>>>(d_A, d_B, d_C, nRow, nCol);
    CUDA_CHECK(cudaGetLastError());

    // 3. D2H Data Transfer
    printf("  Copying result from device to host...\n");
    CUDA_CHECK(cudaMemcpy(h_C_prime, d_C, nBytes, cudaMemcpyDeviceToHost));

    // <<< Record Stop Event >>> (After *last* GPU operation)
    CUDA_CHECK(cudaEventRecord(stop_gpu_total, 0));

    // <<< Synchronize on Stop Event >>>
    // Wait for the GPU to finish executing everything up to the stop event marker
    printf("Waiting for all GPU operations to complete (synchronizing on event)...\n");
    CUDA_CHECK(cudaEventSynchronize(stop_gpu_total));

    // <<< Calculate Elapsed Time >>>
    float millisecondsTotal = 0;
    CUDA_CHECK(cudaEventElapsedTime(&millisecondsTotal, start_gpu_total, stop_gpu_total));

    printf("\n--- Total GPU Performance ---\n");
    printf("Total GPU execution time (H2D + Kernel + D2H): %.3f ms\n", millisecondsTotal);

    // Estimate Effective Bandwidth & GFLOPS based on total time
    if (millisecondsTotal > 0)
    {
        // Calculate effective bandwidth: considers H->D for A & B, and D->H for C
        double totalBytesTransferred = 3.0 * nBytes;
        double secondsElapsed = millisecondsTotal / 1000.0;
        double gibPerSecond = (totalBytesTransferred / secondsElapsed) / (1024.0 * 1024.0 * 1024.0);
        printf("Effective Memory Bandwidth (Combined H2D + Kernel + D2H): %.3f GiB/s\n", gibPerSecond);

        // GFLOPS calculation remains the same (N additions), but the time base is different
        double gflops = (double)N / secondsElapsed / 1e9;
        printf("Overall Performance including transfers: %.3f GFLOPS\n", gflops);
    }

    // Clean up events immediately after use
    CUDA_CHECK(cudaEventDestroy(start_gpu_total));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_total));
    // --- End GPU Work & Timing ---

    // --- Verification ---
    printf("\n--- Verification ---\n");
    checkResult(h_C, h_C_prime, N);

    // --- Cleanup ---
    printf("\nCleaning up resources...\n");
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_prime);

    // Optional: Reset device, but explicit cleanup is often preferred
    // cudaDeviceReset();

    printf("Done\n");
    return 0;
}