#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d "
           "block %d\n",
           iDepth, tid, blockIdx.x);
    // condition to stop recursive execution
    if (iSize == 1)
        return;
    // reduce block size to half
    int nthreads = iSize >> 1;
    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv)
{
    int size = 8;
    int blocksize = 8; // initial block size
    int igrid = 1;

    if (argc > 1)
    {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x,
           block.x);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceReset());

    /*
    Looking at the CUDA code example, I can explain why the block ID for the child grids are all 0 in the output messages.
    The reason is in the nested kernel launch inside the nestedHelloWorld function:
    cpp

    nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);

    When a CUDA kernel is launched, the triple angle brackets <<<grid_dim, block_dim>>> specify the execution configuration. The first parameter (1 in this case) defines the number of blocks in the grid. Since it's explicitly set to 1, all child grid launches will have exactly one block, and therefore all child grid executions will report block 0 in their output messages.
    This is why you see all child grids having block ID 0, even though the parent grid might have multiple blocks (depending on the command line arguments passed to the program).
    If you wanted to change this behavior to have multiple blocks in the child grids, you could modify the kernel launch line to something like:

    nestedHelloWorld<<<some_number_of_blocks, nthreads>>>(nthreads, ++iDepth);
    */
    return 0;
}