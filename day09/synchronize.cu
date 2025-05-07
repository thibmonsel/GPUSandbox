#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <string.h>

#define BLOCK_SIZE_SYNC_TEST 256

/* Example of global synchronization and thread synchronization */

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

__global__ void faulty_kernel(int *out_data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        printf("Faulty Kernel: Thread 0 attempting to write to an invalid address out_data[N+10]. This should cause an error.\n");
        // Deliberate out-of-bounds access (writing to NULL[N+10])
        out_data[N + 10] = 123;
    }
}

void test_device_synchronize_error_reporting()
{
    printf("\n--- Test 1: cudaDeviceSynchronize Error Reporting ---\n");
    const int N = 256;
    dim3 blockDim(256);
    dim3 gridDim(1);

    printf("Launching faulty_kernel asynchronously...\n");
    int *d_actual_data;
    size_t nBytes = (size_t)N * sizeof(int);
    CUDA_CHECK(cudaMalloc((int **)&d_actual_data, nBytes));

    faulty_kernel<<<gridDim, blockDim>>>(d_actual_data, N);

    // Check for immediate launch configuration errors (and clear them)
    cudaError_t launchConfigError = cudaGetLastError();
    if (launchConfigError != cudaSuccess)
    {
        printf("CUDA launch configuration error (cleared by cudaGetLastError): %s\n", cudaGetErrorString(launchConfigError));
    }

    printf("Calling cudaDeviceSynchronize() to wait for completion and catch runtime errors...\n");
    cudaError_t syncError = cudaDeviceSynchronize();

    if (syncError != cudaSuccess)
    {
        printf("SUCCESS: cudaDeviceSynchronize() correctly reported an error from the faulty_kernel: %s\n",
               cudaGetErrorString(syncError));
    }
    else
    {
        // This path might be taken if CUDA_LAUNCH_BLOCKING=1 and launchConfigError already caught the kernel error.
        printf("NOTE: cudaDeviceSynchronize() did NOT report an error. Checking if a previous cudaGetLastError() did \n");
        if (launchConfigError == cudaSuccess)
        {
            printf(" and no prior launch config error was found. This is unexpected for this faulty_kernel setup.\n");
        }
        else
        {
            printf(" a launch config error '%s' was caught earlier. This might be the kernel's runtime error due to sync launch.\n", cudaGetErrorString(launchConfigError));
        }
    }

    // UNCONDITIONALLY clear any host-side error flag that might have been set.
    // This primarily affects the host thread's error variable.
    // It does not fix a corrupted device context if the error was severe.
    cudaError_t finalErrorCheck = cudaGetLastError();
    if (finalErrorCheck != cudaSuccess)
    {
        printf("A host-side sticky error '%s' was present after Test 1 and has now been cleared by the final cudaGetLastError().\n", cudaGetErrorString(finalErrorCheck));
    }
    else
    {
        printf("No further host-side sticky error found by final cudaGetLastError() in Test 1.\n");
    }

    printf("--- Test 1 Finished (Note: The CUDA device/context might be unstable due to the intentional error) ---\n");
}

//-----------------------------------------------------------------------------
// Kernel to demonstrate __syncthreads for block-level synchronization
// Each thread block will perform a partial sum reduction using shared memory.
//-----------------------------------------------------------------------------

__global__ void syncthreads_demo_kernel(const int *input, int *output, int N_per_block)
{
    __shared__ int s_data[BLOCK_SIZE_SYNC_TEST];

    int tid_in_block = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid_in_block < N_per_block)
    {
        s_data[tid_in_block] = input[global_idx];
    }
    else
    {
        s_data[tid_in_block] = 0;
    }

    // ---- System-level: Not applicable here (this is device code) ----
    // ---- Block-level: Wait for all threads in a thread block to reach the same point ----
    __syncthreads(); // Barrier: all threads in the block wait here.
                     // Ensures all s_data writes are complete before any thread proceeds to read.

    // Without __syncthreads() above, a race condition would occur:
    // Some threads might proceed to the reduction phase below
    // and read s_data entries before other threads in the same block
    // have written their values to s_data.

    // Perform a parallel reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid_in_block < s)
        {
            s_data[tid_in_block] += s_data[tid_in_block + s];
        }
        __syncthreads(); // Synchronize after each step of the reduction
                         // to ensure additions are complete before the next level.
    }

    // Thread 0 of each block writes the block's sum to global memory
    if (tid_in_block == 0)
    {
        output[blockIdx.x] = s_data[0];
    }
}

void test_syncthreads_and_shared_memory()
{
    printf("\n--- Test 2: __syncthreads for Block-Level Synchronization & Shared Memory ---\n");

    const int num_blocks = 64;
    const int elements_per_block = BLOCK_SIZE_SYNC_TEST;
    const int total_elements = num_blocks * elements_per_block;

    std::vector<int> h_input(total_elements);
    std::vector<int> h_output_gpu(num_blocks);
    std::vector<int> h_output_cpu(num_blocks);

    for (int i = 0; i < total_elements; ++i)
    {
        h_input[i] = i % 10;
    }

    for (int b = 0; b < num_blocks; ++b)
    {
        int sum = 0;
        for (int i = 0; i < elements_per_block; ++i)
        {
            sum += h_input[b * elements_per_block + i];
        }
        h_output_cpu[b] = sum;
    }

    int *d_input, *d_output;
    size_t input_bytes = total_elements * sizeof(int);
    size_t output_bytes = num_blocks * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE_SYNC_TEST);
    dim3 gridDim(num_blocks);

    printf("Launching syncthreads_demo_kernel (block size: %d, grid size: %d)...\n", blockDim.x, gridDim.x);
    syncthreads_demo_kernel<<<gridDim, blockDim>>>(d_input, d_output, elements_per_block);
    CUDA_CHECK(cudaGetLastError());

    // "System-level: Wait for all work on both the host and the device to complete."
    // This is achieved by cudaDeviceSynchronize().
    // It also acts as a global synchronization point for the kernel just launched.
    printf("Calling cudaDeviceSynchronize() to ensure kernel completion...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < num_blocks; ++i)
    {
        if (h_output_gpu[i] != h_output_cpu[i])
        {
            printf("Mismatch at block %d: GPU_sum = %d, CPU_sum = %d\n", i, h_output_gpu[i], h_output_cpu[i]);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("SUCCESS: __syncthreads demo kernel results verified.\n");
        printf("Example block sum (block 0): %d\n", h_output_gpu[0]);
    }
    else
    {
        printf("FAILURE: __syncthreads demo kernel results do NOT match CPU results.\n");
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    printf("--- Test 2 Finished ---\n");
}

__global__ void first_phase_kernel(int *data, int val, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] = val + idx;
    }
}

__global__ void second_phase_kernel(const int *input_data, int *output_data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // This kernel reads data assumed to be fully written by first_phase_kernel
        output_data[idx] = input_data[idx] * 2;
    }
}

void test_global_synchronization_by_kernel_termination()
{
    printf("\n--- Test 3: Global Synchronization by Kernel Termination ---\n");
    const int N = 1024 * 10;
    size_t nBytes = N * sizeof(int);

    int *d_intermediate_data;
    int *d_final_output;
    std::vector<int> h_final_output(N);

    CUDA_CHECK(cudaMalloc(&d_intermediate_data, nBytes));
    CUDA_CHECK(cudaMalloc(&d_final_output, nBytes));

    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    int initial_val = 100;

    printf("Launching first_phase_kernel...\n");
    first_phase_kernel<<<gridDim, blockDim>>>(d_intermediate_data, initial_val, N);
    // IMPORTANT: At this point, first_phase_kernel is likely still running or queued.
    // If we launched second_phase_kernel immediately, it might read incomplete/incorrect
    // data from d_intermediate_data, leading to a race condition between kernels.

    //  We need to ensure all work from first_phase_kernel is done.
    printf("Calling cudaDeviceSynchronize() to ensure first_phase_kernel completes globally.\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Launching second_phase_kernel which depends on first_phase_kernel's output...\n");
    second_phase_kernel<<<gridDim, blockDim>>>(d_intermediate_data, d_final_output, N);
    CUDA_CHECK(cudaGetLastError());

    printf("Calling cudaDeviceSynchronize() to ensure second_phase_kernel completes.\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_final_output.data(), d_final_output, nBytes, cudaMemcpyDeviceToHost));

    // Verify a few values
    bool success = true;
    for (int i = 0; i < N; i += N / 4)
    { // Check a few sample points
        int expected_intermediate = initial_val + i;
        int expected_final = expected_intermediate * 2;
        if (h_final_output[i] != expected_final)
        {
            printf("Mismatch at index %d: GPU_val = %d, Expected_val = %d\n", i, h_final_output[i], expected_final);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("SUCCESS: Global synchronization test passed. Example output[0]=%d (expected %d).\n", h_final_output[0], (initial_val + 0) * 2);
    }
    else
    {
        printf("FAILURE: Global synchronization test failed.\n");
    }

    CUDA_CHECK(cudaFree(d_intermediate_data));
    CUDA_CHECK(cudaFree(d_final_output));
    printf("--- Test 3 Finished ---\n");
}

int main(int argc, char **argv)
{
    int deviceId;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("Using CUDA Device %d: %s\n", deviceId, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Test 1: Demonstrating cudaDeviceSynchronize reporting errors from async operations
    // We might need to run this first and potentially reset error state if it's sticky.
    test_device_synchronize_error_reporting();

    test_syncthreads_and_shared_memory();

    // Test 3: Demonstrating global synchronization across kernels
    test_global_synchronization_by_kernel_termination();

    printf("\nAll tests completed.\n");
    return 0;
}
