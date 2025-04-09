#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
           "gridDim:(%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}
int main(int argc, char **argv)
{
    // define total data element
    int nElem = 12;
    // define grid and block structure
    /*
    int i(3) is not valid C syntax. This parentheses-based initialization syntax
    (called direct initialization) is a C++ feature and will cause a compilation error in C.
    If you're working with C code, you should always use int i = 3;
    for initialization. If you're seeing int i(3) in code,
    it's either C++ code or it will produce a syntax error when compiled as C.
    */
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();
    // reset device before you leave
    cudaDeviceReset();
    return (0);

    /*
    ACCESS GRID/BLOCK VARIABLES FROM THE HOST AND DEVICE SIDE
s
    It is important to distinguish between the host and device access of grid and block
    variables. For example, using a variable declared as block from the host, you define
    the coordinates and access them as follows:
    block.x, block.y, and block.z
    On the device side, you have pre-initialized, built-in block size variable available as:
    blockDim.x, blockDim.y, and blockDim.z
    In summary, you define variables for grid and block on the host before launching a
    kernel, and access them there with the x, y and z fields of the vector structure from
    the host side. When the kernel is launched, you can use the pre-initialized, built-in
    variables within the kernel.

    blockDim is how many threads you have in a block threadsPerBlock.
    */
}