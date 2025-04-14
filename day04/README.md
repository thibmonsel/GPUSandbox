# Day 4

CUDA matrix add operation. 

```bash
nvprof ./matrixAdd
Matrix size: nRow 16384 nCol 16384
Allocating host memory (4294967296 bytes)...
Initializing host data...
Host data initialization time: 6.66534 sec
Performing matrix addition on host...
Host vector addition time: 8.66570 sec
Allocating Device Memory (1073741824 bytes)...
==308821== NVPROF is profiling process 308821, command: ./matrixAdd

Creating CUDA events for overall GPU timing...
Starting GPU operations and recording events...
  Copying data from host to device (A)...
  Copying data from host to device (B)...
  Launching kernel...
  Copying result from device to host...
Waiting for all GPU operations to complete (synchronizing on event)...

--- Total GPU Performance ---
Total GPU execution time (H2D + Kernel + D2H): 539.108 ms
Effective Memory Bandwidth (Combined H2D + Kernel + D2H): 5.565 GiB/s
Overall Performance including transfers: 0.498 GFLOPS

--- Verification ---
Arrays match.


Cleaning up resources...
Done
==308821== Profiling application: ./matrixAdd
==308821== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.47%  331.09ms         2  165.55ms  165.00ms  166.09ms  [CUDA memcpy HtoD]
                   34.39%  185.23ms         1  185.23ms  185.23ms  185.23ms  [CUDA memcpy DtoH]
                    4.13%  22.268ms         1  22.268ms  22.268ms  22.268ms  matrixAddonDevice(float*, float*, float*, int, int)
      API calls:   78.10%  538.93ms         3  179.64ms  165.09ms  207.65ms  cudaMemcpy
                   21.05%  145.23ms         3  48.409ms  71.270us  145.08ms  cudaMalloc
                    0.62%  4.2535ms       114  37.311us     168ns  2.3777ms  cuDeviceGetAttribute
                    0.18%  1.2359ms         3  411.97us  354.56us  517.64us  cudaFree
                    0.03%  185.97us         2  92.983us  1.4950us  184.47us  cudaEventCreate
                    0.02%  148.90us         1  148.90us  148.90us  148.90us  cudaLaunchKernel
                    0.01%  46.585us         1  46.585us  46.585us  46.585us  cuDeviceGetName
                    0.00%  21.279us         2  10.639us  10.146us  11.133us  cudaEventRecord
                    0.00%  3.3060us         3  1.1020us     228ns  2.7270us  cuDeviceGetCount
                    0.00%  3.2160us         1  3.2160us  3.2160us  3.2160us  cuDeviceGetPCIBusId
                    0.00%  2.2060us         1  2.2060us  2.2060us  2.2060us  cudaEventSynchronize
                    0.00%  1.6050us         1  1.6050us  1.6050us  1.6050us  cudaEventElapsedTime
                    0.00%  1.2140us         2     607ns     266ns     948ns  cudaEventDestroy
                    0.00%  1.1230us         2     561ns     179ns     944ns  cuDeviceGet
                    0.00%     663ns         1     663ns     663ns     663ns  cuDeviceTotalMem
                    0.00%     587ns         1     587ns     587ns     587ns  cuModuleGetLoadingMode
                    0.00%     344ns         1     344ns     344ns     344ns  cudaGetLastError
                    0.00%     336ns         1     336ns     336ns     336ns  cuDeviceGetUuid
```

We see that by utilizing the fact that a matrix is stored linearly in global memory with a row-major approach, adding vectors and matrix can be viewed as the same. In later days, we will use other types of Block and Grid configuration to do matrix multiplications.  

Quote from book p.49 : 

*You are now going to examine this issue in more depth through a matrix addition example. For matrix operations, a natural approach is to use a layout that contains a 2D grid with 2D blocks to organize the threads in your kernel. You will see that a naive approach will not yield the best performance. You are going to learn more about grid and block heuristics using the following layouts for matrix addition: 2D grid with 2D blocks, 1D grid with 1D blocks, 2D grid with 1D blocks*

