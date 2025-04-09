#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    bool ipred = (tid % 2 == 0);
    if (ipred)
    {
        ia = 100.0f;
    }
    if (!ipred)
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);
    // set up data size
    int size = 16 * 1024;
    int blocksize = 64;
    if (argc > 1)
        blocksize = atoi(argv[1]);
    if (argc > 2)
        size = atoi(argv[2]);
    printf("Data size %d ", size);
    // set up execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);
    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);

    cudaMalloc((float **)&d_C, nBytes);
    // run a warmup kernel to remove overhead
    size_t iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("warmup <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);
    // run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);
    // run kernel 3
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);
    // run kernel 3
    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3 <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);
    // // run kernel 4
    // iStart = cpuSecond();
    // mathKernel4<<<grid, block>>>(d_C);
    // cudaDeviceSynchronize();
    // iElaps = cpuSecond() - iStart;
    // printf("mathKernel4 <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x, iElaps);
    // // free gpu memory and reset divece
    // cudaFree(d_C);
    cudaDeviceReset();

    /*
    I can profile with Nsight :

    sudo ncu -o profile --open-in-ui out

    */
    return EXIT_SUCCESS;
}