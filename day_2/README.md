# Day 2 

CUDA vector add operation with streams to gain speed ups.

```bash
./vectorAddWithStreams 
Starting Vector Add with CUDA Streams...
Configuration:
Vector elements (N): 1048576 (1.00 Million)
Vector size: 4194304 bytes (4.00 MB)
CUDA Streams: 2
Threads per Block: 256
Allocating Pinned Host Memory...
Initializing host data...
Host data initialization time: 0.02551 sec
Performing vector addition on host...
Host vector addition time: 0.00215 sec
Allocating Device Memory...
Creating 2 CUDA streams...
Creating CUDA events for timing...

Starting GPU execution with 2 streams...
  Stream 0: Offset=0, Elements=524288, Size=2048.00 KB
  Stream 1: Offset=524288, Elements=524288, Size=2048.00 KB
Synchronizing device...
Device synchronization complete.

--- GPU Performance ---
Total GPU execution time (H2D Copy + Kernel + D2H Copy): 1.030 ms
Effective Memory Bandwidth (Combined H2D + D2H): 11.376 GiB/s
Kernel Estimated Performance: 1.018 GFLOPS

--- Verification ---
Arrays match.

Cleaning up resources...
Done
```

Profiling of the code with `nvprof` with `4` streams. (In this example increasing the number of streams doesn't necessarily help)

```bash
==479561== Profiling application: ./vectorAddWithStreams
==479561== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.68%  722.53us         4  180.63us  168.38us  205.82us  [CUDA memcpy HtoD]
                   31.37%  373.47us         2  186.74us  161.25us  212.22us  [CUDA memcpy DtoH]
                    7.95%  94.688us         2  47.344us  46.528us  48.160us  vectorAddOnDevice(float*, float*, float*, int)
      API calls:   95.16%  138.83ms         4  34.708ms  813.30us  136.36ms  cudaMallocHost
                    2.89%  4.2092ms       114  36.923us     166ns  2.3861ms  cuDeviceGetAttribute
                    0.86%  1.2508ms         4  312.69us  295.87us  357.55us  cudaFreeHost
                    0.41%  604.98us         1  604.98us  604.98us  604.98us  cudaDeviceSynchronize
                    0.26%  386.54us         2  193.27us  5.1460us  381.39us  cudaLaunchKernel
                    0.19%  281.12us         3  93.706us  38.593us  185.77us  cudaMalloc
                    0.11%  159.07us         3  53.023us  31.991us  94.109us  cudaFree
                    0.03%  46.514us         1  46.514us  46.514us  46.514us  cuDeviceGetName
                    0.02%  36.342us         6  6.0570us  1.8570us  17.350us  cudaMemcpyAsync
                    0.02%  26.458us         2  13.229us  1.9590us  24.499us  cudaStreamCreate
                    0.02%  23.972us         2  11.986us  6.2890us  17.683us  cudaEventRecord
                    0.01%  13.156us         2  6.5780us  1.8480us  11.308us  cudaStreamDestroy
                    0.00%  5.0990us         2  2.5490us     289ns  4.8100us  cudaEventCreate
                    0.00%  4.7500us         1  4.7500us  4.7500us  4.7500us  cuDeviceGetPCIBusId
                    0.00%  2.9560us         3     985ns     214ns  2.4780us  cuDeviceGetCount
                    0.00%  1.8460us         1  1.8460us  1.8460us  1.8460us  cudaEventSynchronize
                    0.00%  1.8350us         2     917ns     184ns  1.6510us  cudaEventDestroy
                    0.00%  1.1380us         1  1.1380us  1.1380us  1.1380us  cudaEventElapsedTime
                    0.00%     934ns         2     467ns     168ns     766ns  cuDeviceGet
                    0.00%     579ns         1     579ns     579ns     579ns  cuDeviceTotalMem
                    0.00%     426ns         1     426ns     426ns     426ns  cuModuleGetLoadingMode
                    0.00%     381ns         2     190ns     100ns     281ns  cudaGetLastError
                    0.00%     329ns         1     329ns     329ns     329ns  cuDeviceGetUuid
```

In this case here we see that overall implementation with streams has a 2x speed up compared to day 1's implementation. This is due to having asynchronous data loading schema that is possible thanks to CUDA streams. What you see also is that `vectorAddOnDevice` time hasn't changed much !