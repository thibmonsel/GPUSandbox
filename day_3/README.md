# Day 3

CUDA vector add operation with an optimized CUDA type `float4` to decrease `vectorAdd` operation.

```bash
nvprof ./vectorAddOptimized 
Starting Vector Add with CUDA Streams (Optimized Kernel)...
Configuration:
Vector elements (N): 1048576 (1.00 Million)
Vector size: 4194304 bytes (4.00 MB)
CUDA Streams: 4
Threads per Block: 256
Using float4 vectorized kernel.
Allocating Pinned Host Memory...
==752241== NVPROF is profiling process 752241, command: ./vectorAddOptimized
Initializing host data...
Host data initialization time: 0.03316 sec
Performing vector addition on host...
Host vector addition time: 0.00285 sec
Allocating Device Memory...
Creating 4 CUDA streams...
Creating CUDA events for timing...

Starting GPU execution with 4 streams (Optimized Kernel)...
Stream 0: Offset(floats)=0, Elements(floats)=262144 (float4=65536), Size=1024.00 KB
Stream 1: Offset(floats)=262144, Elements(floats)=262144 (float4=65536), Size=1024.00 KB
Stream 2: Offset(floats)=524288, Elements(floats)=262144 (float4=65536), Size=1024.00 KB
Stream 3: Offset(floats)=786432, Elements(floats)=262144 (float4=65536), Size=1024.00 KB
Synchronizing device...
Device synchronization complete.

--- GPU Performance ---
Total GPU execution time (H2D Copy + Kernel + D2H Copy): 1.060 ms
Effective Memory Bandwidth (Combined H2D + D2H): 11.054 GiB/s
Kernel Estimated Performance: 0.989 GFLOPS

--- Verification ---
Arrays match.

Cleaning up resources...
Done
==752241== Profiling application: ./vectorAddOptimized
==752241== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.84%  801.28us         8  100.16us  87.104us  118.85us  [CUDA memcpy HtoD]
                   29.71%  418.91us         4  104.73us  80.800us  116.03us  [CUDA memcpy DtoH]
                   13.45%  189.63us         4  47.408us  46.592us  47.840us  vectorAddOptimized(float4 const *, float4 const *, float4*, int)
      API calls:   95.22%  142.71ms         4  35.677ms  869.28us  139.86ms  cudaMallocHost
                    2.67%  4.0073ms       114  35.151us      41ns  2.2148ms  cuDeviceGetAttribute
                    0.97%  1.4526ms         4  363.14us  338.43us  374.21us  cudaFreeHost
                    0.48%  715.56us         1  715.56us  715.56us  715.56us  cudaDeviceSynchronize
                    0.21%  312.49us         3  104.16us  52.404us  197.46us  cudaMalloc
                    0.17%  252.08us         4  63.021us  4.1150us  220.36us  cudaLaunchKernel
                    0.14%  210.52us         3  70.171us  36.667us  123.93us  cudaFree
                    0.04%  58.916us        12  4.9090us  2.1760us  20.291us  cudaMemcpyAsync
                    0.03%  40.169us         1  40.169us  40.169us  40.169us  cuDeviceGetName
                    0.02%  34.126us         4  8.5310us  1.2190us  28.798us  cudaStreamCreate
                    0.02%  24.514us         2  12.257us  4.6830us  19.831us  cudaEventRecord
                    0.01%  21.996us         2  10.998us     428ns  21.568us  cudaEventCreate
                    0.01%  17.430us         4  4.3570us  1.7480us  12.026us  cudaStreamDestroy
                    0.00%  5.1490us         1  5.1490us  5.1490us  5.1490us  cuDeviceGetPCIBusId
                    0.00%  3.3930us         3  1.1310us     234ns  2.8440us  cuDeviceGetCount
                    0.00%  2.9850us         2  1.4920us     236ns  2.7490us  cudaEventDestroy
                    0.00%  2.1720us         1  2.1720us  2.1720us  2.1720us  cudaEventSynchronize
                    0.00%  1.8180us         1  1.8180us  1.8180us  1.8180us  cudaEventElapsedTime
                    0.00%     937ns         2     468ns     170ns     767ns  cuDeviceGet
                    0.00%     603ns         1     603ns     603ns     603ns  cuDeviceTotalMem
                    0.00%     519ns         4     129ns      70ns     287ns  cudaGetLastError
                    0.00%     417ns         1     417ns     417ns     417ns  cuModuleGetLoadingMode
                    0.00%      83ns         1      83ns      83ns      83ns  cuDeviceGetUuid
```

Here thanks to the `float4` CUDA type, 20% of the execution of vector operation was shed off.