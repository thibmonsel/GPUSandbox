# Day 2 

CUDA vector add operation with streams to gain speed ups.

```
 ./vectorAddWithStreams 
Starting... 
Vector size: 1048576 that will be of size 4194304 bytes (sizeof(float)=4) 
Base chunk size (approx): 65536 elements
Init time for vectors: 0.02467 sec
Vector addition on the host: 0.00260 sec
Starting GPU vector add with streams...
Stream 0: offset=0, elements=65536, bytes=256.00 KB
Stream 1: offset=65536, elements=65536, bytes=256.00 KB
Stream 2: offset=131072, elements=65536, bytes=256.00 KB
Stream 3: offset=196608, elements=65536, bytes=256.00 KB
Stream 4: offset=262144, elements=65536, bytes=256.00 KB
Stream 5: offset=327680, elements=65536, bytes=256.00 KB
Stream 6: offset=393216, elements=65536, bytes=256.00 KB
Stream 7: offset=458752, elements=65536, bytes=256.00 KB
Stream 8: offset=524288, elements=65536, bytes=256.00 KB
Stream 9: offset=589824, elements=65536, bytes=256.00 KB
Stream 10: offset=655360, elements=65536, bytes=256.00 KB
Stream 11: offset=720896, elements=65536, bytes=256.00 KB
Stream 12: offset=786432, elements=65536, bytes=256.00 KB
Stream 13: offset=851968, elements=65536, bytes=256.00 KB
Stream 14: offset=917504, elements=65536, bytes=256.00 KB
Stream 15: offset=983040, elements=65536, bytes=256.00 KB
GPU execution time (streams): 1.006 ms
Effective Bandwidth: 11.65 GB/s
Arrays match.
Done
```

Profiling of the code with `nvprof`.

```
==401568== Profiling application: ./vectorAddWithStreams
==401568== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.84%  1.3887ms        32  43.397us  23.648us  77.280us  [CUDA memcpy HtoD]
                   33.73%  757.50us        16  47.343us  28.448us  98.080us  [CUDA memcpy DtoH]
                    4.43%  99.587us        16  6.2240us  5.6960us  6.8490us  vectorAddOnDevice(float*, float*, float*, int)
      API calls:   59.04%  141.58ms         4  35.394ms  823.48us  138.93ms  cudaMallocHost
                   38.41%  92.092ms         1  92.092ms  92.092ms  92.092ms  cudaDeviceReset
                    1.66%  3.9906ms       114  35.005us     144ns  2.2124ms  cuDeviceGetAttribute
                    0.53%  1.2759ms         1  1.2759ms  1.2759ms  1.2759ms  cudaDeviceSynchronize
                    0.11%  257.82us        16  16.113us  2.6690us  178.46us  cudaLaunchKernel
                    0.09%  208.27us         3  69.424us  32.653us  134.99us  cudaMalloc
                    0.08%  183.70us        48  3.8270us  1.5360us  19.915us  cudaMemcpyAsync
                    0.04%  93.483us        16  5.8420us     793ns  52.527us  cudaStreamCreate
                    0.02%  55.790us         1  55.790us  55.790us  55.790us  cuDeviceGetName
                    0.01%  24.992us         2  12.496us  4.4600us  20.532us  cudaEventRecord
                    0.00%  6.0580us         1  6.0580us  6.0580us  6.0580us  cuDeviceGetPCIBusId
                    0.00%  5.5950us         2  2.7970us     248ns  5.3470us  cudaEventCreate
                    0.00%  2.9370us         3     979ns     159ns  2.3860us  cuDeviceGetCount
                    0.00%  2.3920us         1  2.3920us  2.3920us  2.3920us  cudaEventSynchronize
                    0.00%  1.3440us         1  1.3440us  1.3440us  1.3440us  cudaEventElapsedTime
                    0.00%     972ns         2     486ns     168ns     804ns  cuDeviceGet
                    0.00%     906ns        16      56ns      37ns     262ns  cudaGetLastError
                    0.00%     899ns         1     899ns     899ns     899ns  cuDeviceTotalMem
                    0.00%     418ns         1     418ns     418ns     418ns  cuModuleGetLoadingMode
                    0.00%     331ns         1     331ns     331ns     331ns  cuDeviceGetUuid
```

In this case here we see that `vectorAddOnDevice` implementation with streams has a 2x speed up compared to `vectorAddOnDevice`'s day 1.  