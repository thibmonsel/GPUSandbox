/*
ORGANIZING PARALLEL THREADS

You are now going to examine this issue in more depth through a matrix addition example. For
matrix operations, a natural approach is to use a layout that contains a 2D grid with 2D blocks to
organize the threads in your kernel. You will see that a naive approach will not yield the best perfor-
mance.

*/

/*
Indexing Matrices with Blocks and Threads

Typically, a matrix is stored linearly in global memory with a row-major approach. Figure 2-9 illus-
trates a small case for an 8 x 6 matrix.

Row 0 - Row 1 - ... - Row 5 where size of Row i is 8.
*/

/*
In a matrix addition kernel, a thread is usually assigned one data element to process. Accessing the
assigned data from global memory using block and thread index is the first issue you need to solve.

Typically, there are three kinds of indices for a 2D case you need to manage:

➤ Thread and block index
➤ Coordinate of a given point in the matrix
➤ Offset in linear global memory


For a given thread, you can obtain the offset in global memory from the block and thread index by
first mapping the thread and block index to coordinates in the matrix, then mapping those matrix
coordinates to a global memory location.

In the first step, you can map the thread and block index to the coordinate of a matrix with the
following formula:

ix = threadIdx.x + blockIdx.x * blockDim.x
iy = threadIdx.y + blockIdx.y * blockDim.y

In the second step, you can map a matrix coordinate to a global memory location/index with the
following formula:

idx = iy * nx + ix

*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(-10 * error);                                                 \
        }                                                                      \
    }
void initialInt(int *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = i;
    }
}
void printMatrix(int *C, const int nx, const int ny)
{
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}
__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
           "global index %2d ival %2d\n",
           threadIdx.x, threadIdx.y, blockIdx.x,
           blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    // malloc host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);
    // initialize host matrix with integer
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);
    // malloc device memory
    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);
    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    // set up execution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    // invoke the kernel
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();
    // free host and devide memory
    cudaFree(d_MatA);
    free(h_A);
    // reset device
    cudaDeviceReset();
    return (0);
}