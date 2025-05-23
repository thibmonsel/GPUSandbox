# Day 06

CUDA matrix addition using a 2D block and 2D grid configuration, where each block is responsible for a matrix chunk.

```bash
nvprof ./matrixAdd3
Matrix size: nx 16384 ny 16384
Allocating host memory (4294967296 bytes)...
Initializing host data...
Host data initialization time: 7.46754 sec
Performing matrix addition on host...
Host vector addition time: 8.49490 sec
Allocating Device Memory (1073741824 bytes)...
==123765== NVPROF is profiling process 123765, command: ./matrixAdd3

Creating CUDA events for overall GPU timing...
Starting GPU operations and recording events...
  Copying data from host to device (A)...
  Copying data from host to device (B)...
  Launching kernel...
  Copying result from device to host...
Waiting for all GPU operations to complete (synchronizing on event)...

--- Total GPU Performance ---
Total GPU execution time (H2D + Kernel + D2H): 624.810 ms
Effective Memory Bandwidth (Combined H2D + Kernel + D2H): 1.600 GiB/s
Overall Performance including transfers: 0.430 GFLOPS

--- Verification ---
Arrays match.


Cleaning up resources...
Done
==123765== Profiling application: ./matrixAdd3
==123765== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.75%  391.68ms         2  195.84ms  175.23ms  216.44ms  [CUDA memcpy HtoD]
                   31.10%  194.11ms         1  194.11ms  194.11ms  194.11ms  [CUDA memcpy DtoH]
                    6.15%  38.404ms         1  38.404ms  38.404ms  38.404ms  matrixAddonDevice(float*, float*, float*, int, int)
      API calls:   80.36%  624.57ms         3  208.19ms  175.33ms  232.69ms  cudaMemcpy
                   18.95%  147.29ms         3  49.097ms  85.376us  147.12ms  cudaMalloc
                    0.52%  4.0073ms       114  35.152us      49ns  2.2523ms  cuDeviceGetAttribute
                    0.14%  1.1083ms         3  369.44us  305.69us  490.44us  cudaFree
                    0.02%  193.61us         1  193.61us  193.61us  193.61us  cudaLaunchKernel
                    0.00%  38.144us         2  19.072us  10.565us  27.579us  cudaEventRecord
                    0.00%  22.110us         1  22.110us  22.110us  22.110us  cuDeviceGetName
                    0.00%  8.1860us         2  4.0930us     413ns  7.7730us  cudaEventCreate
                    0.00%  4.1890us         1  4.1890us  4.1890us  4.1890us  cuDeviceGetPCIBusId
                    0.00%  2.0410us         1  2.0410us  2.0410us  2.0410us  cudaEventSynchronize
                    0.00%  1.9940us         1  1.9940us  1.9940us  1.9940us  cudaEventElapsedTime
                    0.00%  1.4460us         3     482ns      50ns  1.3150us  cuDeviceGetCount
                    0.00%  1.4270us         2     713ns     373ns  1.0540us  cudaEventDestroy
                    0.00%     433ns         1     433ns     433ns     433ns  cuDeviceTotalMem
                    0.00%     401ns         2     200ns      51ns     350ns  cuDeviceGet
                    0.00%     311ns         1     311ns     311ns     311ns  cudaGetLastError
                    0.00%     207ns         1     207ns     207ns     207ns  cuModuleGetLoadingMode
                    0.00%      92ns         1      92ns      92ns      92ns  cuDeviceGetUuid
```


Compared to day05's implementation, `matrixAddonDevice` is slower (we used the same number of threads per block i.e. 1024.). Threads that are in the same block have access to the same shared memory region (SMEM).
