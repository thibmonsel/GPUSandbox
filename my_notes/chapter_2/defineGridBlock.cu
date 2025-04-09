#include <cuda_runtime.h>
#include <stdio.h>
int main(int argc, char **argv)
{
    // define total data elements
    int nElem = 1024;
    // define grid and block structure
    dim3 block(1024);
    dim3 grid((nElem - 1) / block.x + 1);
    printf("grid.x %d %d %d block.x %d \n", grid.x, grid.y, grid.z, block.x);
    // reset block
    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d %d %d block.x %d \n", grid.x, grid.y, grid.z, block.x);
    // reset block
    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d %d %d block.x %d \n", grid.x, grid.y, grid.z, block.x);
    // reset block
    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d %d %d block.x %d \n", grid.x, grid.y, grid.z, block.x);
    // reset device before you leave
    cudaDeviceReset();
    return (0);
}