# Day 2 

CUDA vector add operation with streams to gain speed ups. Profiling of the code with `nvprof` with `4` streams. (In this example increasing the number of streams doesn't necessarily help)


```bash
nvprof ./vectorAddWithStreams
Starting Vector Add with CUDA Streams...
Configuration:
Vector elements (N): 1048576 (1.00 Million)
Vector size: 4194304 bytes (4.00 MB)
CUDA Streams: 2
Threads per Block: 256
Allocating Pinned Host Memory...
==752954== NVPROF is profiling process 752954, command: ./vectorAddWithStreams
Initializing host data...
Host data initialization time: 0.03531 sec
Performing vector addition on host...
Host vector addition time: 0.00267 sec
Allocating Device Memory...
Creating 2 CUDA streams...
Creating CUDA events for timing...

Starting GPU execution with 2 streams...
  Stream 0: Offset=0, Elements=524288, Size=2048.00 KB
  Stream 1: Offset=524288, Elements=524288, Size=2048.00 KB
Synchronizing device...
Device synchronization complete.

--- GPU Performance ---
Total GPU execution time (H2D Copy + Kernel + D2H Copy): 1.155 ms
Effective Memory Bandwidth (Combined H2D + D2H): 10.145 GiB/s
Kernel Estimated Performance: 0.908 GFLOPS

--- Verification ---
Arrays match.

Cleaning up resources...
Done
==752954== Profiling application: ./vectorAddWithStreams
==752954== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.11%  764.07us         4  191.02us  170.59us  213.25us  [CUDA memcpy HtoD]
                   28.19%  390.79us         2  195.39us  161.34us  229.44us  [CUDA memcpy DtoH]
                   16.70%  231.52us         2  115.76us  114.59us  116.93us  vectorAddOnDevice(float*, float*, float*, int)
      API calls:   95.18%  150.16ms         4  37.540ms  1.0699ms  146.88ms  cudaMallocHost
                    2.78%  4.3863ms       114  38.476us     168ns  2.2019ms  cuDeviceGetAttribute
                    0.94%  1.4880ms         4  371.99us  343.32us  411.73us  cudaFreeHost
                    0.43%  682.93us         1  682.93us  682.93us  682.93us  cudaDeviceSynchronize
                    0.26%  408.57us         2  204.29us  5.6740us  402.90us  cudaLaunchKernel
                    0.18%  276.71us         3  92.235us  39.325us  189.91us  cudaMalloc
                    0.11%  179.08us         3  59.694us  33.558us  108.69us  cudaFree
                    0.03%  44.691us         1  44.691us  44.691us  44.691us  cuDeviceGetName
                    0.02%  37.157us         6  6.1920us  2.2110us  18.350us  cudaMemcpyAsync
                    0.02%  29.492us         2  14.746us  1.3040us  28.188us  cudaStreamCreate
                    0.02%  23.795us         2  11.897us  4.7110us  19.084us  cudaEventRecord
                    0.01%  14.362us         2  7.1810us  1.9840us  12.378us  cudaStreamDestroy
                    0.00%  7.2630us         2  3.6310us  1.3270us  5.9360us  cudaEventCreate
                    0.00%  5.0370us         1  5.0370us  5.0370us  5.0370us  cuDeviceGetPCIBusId
                    0.00%  3.9590us         3  1.3190us     164ns  3.4550us  cuDeviceGetCount
                    0.00%  2.2760us         1  2.2760us  2.2760us  2.2760us  cudaEventSynchronize
                    0.00%  2.2050us         2  1.1020us     366ns  1.8390us  cudaEventDestroy
                    0.00%  1.5830us         1  1.5830us  1.5830us  1.5830us  cudaEventElapsedTime
                    0.00%     994ns         2     497ns     173ns     821ns  cuDeviceGet
                    0.00%     834ns         1     834ns     834ns     834ns  cuDeviceTotalMem
                    0.00%     604ns         1     604ns     604ns     604ns  cuDeviceGetUuid
                    0.00%     460ns         1     460ns     460ns     460ns  cuModuleGetLoadingMode
                    0.00%     457ns         2     228ns      72ns     385ns  cudaGetLastError
```

In this case here we see that overall implementation with streams has a 2x speed up compared to day 1's implementation. This is due to having asynchronous data loading schema that is possible thanks to CUDA streams. What you see also is that `vectorAddOnDevice` time hasn't changed much !

Quote p.148-49 :

*The GPU cannot safely access data in pageable host memory because it has no control over when
the host operating system may choose to physically move that data. When transferring data from
pageable host memory to device memory, the CUDA driver first allocates temporary page-locked 
or pinned host memory, copies the source host data to pinned memory, and then transfers the data
from pinned memory to device memory, as illustrated on the left side of Figure 4-4*.

RULES : 

**MEMORY TRANSFER BETWEEN THE HOST AND DEVICE**

**Pinned memory is more expensive to allocate and deallocate than pageable memory, 
but it provides higher transfer throughput for large data transfers.
The speedup achieved when using pinned memory relative to pageable memory
depends on device compute capability. For example, on Fermi devices it is generally
beneficial to use pinned memory when transferring more than 10 MB of data.
Batching many small transfers into one larger transfer improves performance
because it reduces per-transfer overhead.
Data transfers between the host and device can sometimes be overlapped with
kernel execution. You will learn more about this topic in Chapter 6, “Streams and
Concurrency.” You should either minimize or overlap data transfers between the
host and device whenever possible.**