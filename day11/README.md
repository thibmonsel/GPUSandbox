# Day11

Rearranging thread indices to reduce warp divergence. We saw in day10 that for the parallel reduction operation, less and less threads are active during the computation.
The idea is to change which threads are active, not necessarily where they read from or write to for the final partial sums. Instead of having active threads spread out (0, 4, 8,...), you want a compact block of active threads (0, 1, 2, 3,...).

In this improved pattern, in the first step, threads 0 to (N/2) - 1 are active. These threads are contiguous. If N/2 is greater than or equal to the warp size (32), then many warps will be fully active. As N decreases, fewer warps are used, but the active warps are mostly full until the very end.

```bash
Global Memory:     3   1   7   0   4   1   6   3
                   ↓       ↓       ↓       ↓
Thread ID 0     → (3+1)=4
Thread ID 1             → (7+0)=7
Thread ID 2                     → (4+1)=5
Thread ID 3                             → (6+3)=9
                                       ↓       ↓
                             Thread ID 0   → (4+7)=11
                             Thread ID 1           → (5+9)=14
                                                  ↓       ↓
                                        Thread ID 0   → (11+14)=25

Result:             25   _   _   _   _   _   _   _
```


Profiling `parallelReduce2.cu` that has less divergence : 

```bash
nvprof ./parallelReduce2
==761670== NVPROF is profiling process 761670, command: ./parallelReduce2
Using CUDA Device: NVIDIA T600 Laptop GPU
Compute Capability: 7.5
Max Threads Per Block: 1024
Max Threads Per SM: 1024
Warp Size: 32
Total Global Memory: 3718.94 MiB

--- Testing Parallel Reduction (Neighbor-Paired) ---

Problem Size N = 1 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -100
GPU Sum (actual):   -100
Verification: PASS
Elapsed Time: 0.018 ms
Performance:  0.000 GigaElements/s
Effective Bandwidth (kernel part): 0.000 GiB/s

Problem Size N = 128 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4672
GPU Sum (actual):   -4672
Verification: PASS
Elapsed Time: 0.017 ms
Performance:  0.007 GigaElements/s
Effective Bandwidth (kernel part): 0.028 GiB/s

Problem Size N = 256 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4160
GPU Sum (actual):   -4160
Verification: PASS
Elapsed Time: 0.017 ms
Performance:  0.015 GigaElements/s
Effective Bandwidth (kernel part): 0.056 GiB/s

Problem Size N = 257 elements
Configuration: 2 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.01 KiB
CPU Sum (expected): -4204
GPU Sum (actual):   -4204
Verification: PASS
Elapsed Time: 0.017 ms
Performance:  0.015 GigaElements/s
Effective Bandwidth (kernel part): 0.056 GiB/s

Problem Size N = 1048576 elements
Configuration: 4096 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -526400
GPU Sum (actual):   -526400
Verification: PASS
Elapsed Time: 0.784 ms
Performance:  1.337 GigaElements/s
Effective Bandwidth (kernel part): 5.001 GiB/s

Problem Size N = 1048589 elements
Configuration: 4097 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -525334
GPU Sum (actual):   -525334
Verification: PASS
Elapsed Time: 0.781 ms
Performance:  1.342 GigaElements/s
Effective Bandwidth (kernel part): 5.018 GiB/s

Problem Size N = 4194304 elements
Configuration: 16384 blocks, 256 threads/block
Input data: 16.00 MiB, Output (block sums): 64.00 KiB
CPU Sum (expected): -2102144
GPU Sum (actual):   -2102144
Verification: PASS
Elapsed Time: 3.069 ms
Performance:  1.367 GigaElements/s
Effective Bandwidth (kernel part): 5.111 GiB/s
-----------------------------------------------------

Done.
==761670== Profiling application: ./parallelReduce2
==761670== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.10%  9.3224ms        14  665.88us  10.368us  3.0648ms  reduceNeighboredLess(int*, int*, unsigned int)
                   46.64%  8.1888ms        14  584.91us     480ns  3.5283ms  [CUDA memcpy HtoD]
                    0.15%  26.784us         7  3.8260us  1.0240us  8.1280us  [CUDA memset]
                    0.11%  18.944us         7  2.7060us  1.4400us  7.1040us  [CUDA memcpy DtoH]
      API calls:   53.85%  147.60ms         2  73.802ms     460ns  147.60ms  cudaEventCreate
                   36.03%  98.758ms         1  98.758ms  98.758ms  98.758ms  cudaDeviceReset
                    3.13%  8.5776ms        21  408.46us  1.9170us  3.6960ms  cudaMemcpy
                    1.81%  4.9478ms         7  706.83us  10.861us  3.1319ms  cudaEventSynchronize
                    1.79%  4.9085ms         7  701.22us  12.819us  3.1165ms  cudaDeviceSynchronize
                    1.47%  4.0193ms       114  35.256us     166ns  2.2064ms  cuDeviceGetAttribute
                    1.28%  3.5096ms         1  3.5096ms  3.5096ms  3.5096ms  cudaGetDeviceProperties
                    0.24%  664.93us        14  47.494us  1.5210us  199.26us  cudaFree
                    0.24%  653.48us        14  46.676us  1.2260us  208.03us  cudaMalloc
                    0.09%  260.38us        14  18.598us  2.3270us  122.96us  cudaLaunchKernel
                    0.03%  77.937us         7  11.133us  2.6930us  38.528us  cudaMemset
                    0.02%  55.083us        14  3.9340us  1.8980us  12.232us  cudaEventRecord
                    0.01%  37.352us         1  37.352us  37.352us  37.352us  cuDeviceGetName
                    0.00%  7.4410us         7  1.0630us     544ns  2.0580us  cudaEventElapsedTime
                    0.00%  6.2320us         1  6.2320us  6.2320us  6.2320us  cuDeviceGetPCIBusId
                    0.00%  4.2850us         1  4.2850us  4.2850us  4.2850us  cudaGetDevice
                    0.00%  3.5240us         3  1.1740us     170ns  2.9870us  cuDeviceGetCount
                    0.00%  2.5370us         2  1.2680us     351ns  2.1860us  cudaEventDestroy
                    0.00%  1.8410us        14     131ns      39ns     360ns  cudaGetLastError
                    0.00%     852ns         2     426ns     184ns     668ns  cuDeviceGet
                    0.00%     672ns         1     672ns     672ns     672ns  cuDeviceTotalMem
                    0.00%     459ns         1     459ns     459ns     459ns  cuModuleGetLoadingMode
                    0.00%     282ns         1     282ns     282ns     282ns  cuDeviceGetUuid
```

Compared to day10 we see that the parallel reduction operation goes from `15.888ms` to `9.3224ms`, thus showing the reduction of warp divergence.
As seen in small case where $N = 256$, we see that performance are similar since the number of warps scheduled and active are very small.