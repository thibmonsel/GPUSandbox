# Day10

Visual example of a [Parallel reduction](https://en.wikipedia.org/wiki/Reduction_operator) with a Neighbored pair method: Elements are paired with their immediate neighbor. In this implementation, a thread takes two adjacent elements to produce one partial sum at each step. For an array with N elements, this implementation requires $N − 1$ sums and $\log_2(N)$ steps.

```bash
Global Memory:     3   1   7   0   4   1   6   3
                   ↓       ↓       ↓       ↓
Thread ID 0     → (3+1)=4
Thread ID 2             → (7+0)=7
Thread ID 4                     → (4+1)=5
Thread ID 6                             → (6+3)=9
                                       ↓       ↓
                             Thread ID 0   → (4+7)=11
                             Thread ID 4           → (5+9)=14
                                                  ↓       ↓
                                        Thread ID 0   → (11+14)=25

Result:             25   _   _   _   _   _   _   _
```


Profiling `parallelReduce.cu` : 

```bash
nvprof ./parallelReduce
==761285== NVPROF is profiling process 761285, command: ./parallelReduce
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
Elapsed Time: 0.022 ms
Performance:  0.000 GigaElements/s
Effective Bandwidth (kernel part): 0.000 GiB/s

Problem Size N = 128 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4672
GPU Sum (actual):   -4672
Verification: PASS
Elapsed Time: 0.021 ms
Performance:  0.006 GigaElements/s
Effective Bandwidth (kernel part): 0.023 GiB/s

Problem Size N = 256 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4160
GPU Sum (actual):   -4160
Verification: PASS
Elapsed Time: 0.021 ms
Performance:  0.012 GigaElements/s
Effective Bandwidth (kernel part): 0.045 GiB/s

Problem Size N = 257 elements
Configuration: 2 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.01 KiB
CPU Sum (expected): -4204
GPU Sum (actual):   -4204
Verification: PASS
Elapsed Time: 0.022 ms
Performance:  0.012 GigaElements/s
Effective Bandwidth (kernel part): 0.044 GiB/s

Problem Size N = 1048576 elements
Configuration: 4096 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -526400
GPU Sum (actual):   -526400
Verification: PASS
Elapsed Time: 1.331 ms
Performance:  0.788 GigaElements/s
Effective Bandwidth (kernel part): 2.946 GiB/s

Problem Size N = 1048589 elements
Configuration: 4097 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -525334
GPU Sum (actual):   -525334
Verification: PASS
Elapsed Time: 1.324 ms
Performance:  0.792 GigaElements/s
Effective Bandwidth (kernel part): 2.961 GiB/s

Problem Size N = 4194304 elements
Configuration: 16384 blocks, 256 threads/block
Input data: 16.00 MiB, Output (block sums): 64.00 KiB
CPU Sum (expected): -2102144
GPU Sum (actual):   -2102144
Verification: PASS
Elapsed Time: 5.252 ms
Performance:  0.799 GigaElements/s
Effective Bandwidth (kernel part): 2.987 GiB/s
-----------------------------------------------------

Done.
==761285== Profiling application: ../day10/vectorReduce
==761285== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.94%  15.888ms        14  1.1348ms  15.199us  5.2410ms  reduceNeighbored(int*, int*, unsigned int)
                   34.87%  8.5318ms        14  609.41us     512ns  4.4911ms  [CUDA memcpy HtoD]
                    0.11%  26.560us         7  3.7940us     992ns  8.1280us  [CUDA memset]
                    0.08%  19.200us         7  2.7420us  1.4400us  7.3920us  [CUDA memcpy DtoH]
      API calls:   52.17%  147.70ms         2  73.848ms     509ns  147.70ms  cudaEventCreate
                   34.90%  98.818ms         1  98.818ms  98.818ms  98.818ms  cudaDeviceReset
                    3.12%  8.8250ms        21  420.24us  1.9750us  4.6330ms  cudaMemcpy
                    2.91%  8.2424ms         7  1.1775ms  17.254us  5.3188ms  cudaEventSynchronize
                    2.88%  8.1408ms         7  1.1630ms  17.295us  5.2841ms  cudaDeviceSynchronize
                    1.39%  3.9495ms       114  34.644us      43ns  2.1537ms  cuDeviceGetAttribute
                    1.17%  3.3061ms         1  3.3061ms  3.3061ms  3.3061ms  cudaGetDeviceProperties
                    0.85%  2.4123ms        14  172.31us  2.4160us  2.2614ms  cudaLaunchKernel
                    0.28%  787.59us        14  56.256us  1.7180us  211.39us  cudaFree
                    0.26%  740.66us        14  52.904us  1.4030us  223.93us  cudaMalloc
                    0.03%  93.972us         7  13.424us  2.7140us  46.329us  cudaMemset
                    0.02%  48.426us        14  3.4590us  2.0780us  7.2500us  cudaEventRecord
                    0.01%  41.154us         1  41.154us  41.154us  41.154us  cuDeviceGetName
                    0.00%  7.3130us         7  1.0440us     564ns  1.5330us  cudaEventElapsedTime
                    0.00%  6.0970us         1  6.0970us  6.0970us  6.0970us  cuDeviceGetPCIBusId
                    0.00%  3.3200us         3  1.1060us     229ns  2.7170us  cuDeviceGetCount
                    0.00%  3.0620us        14     218ns      39ns     959ns  cudaGetLastError
                    0.00%  2.2210us         2  1.1100us     371ns  1.8500us  cudaEventDestroy
                    0.00%  1.6080us         1  1.6080us  1.6080us  1.6080us  cudaGetDevice
                    0.00%     738ns         1     738ns     738ns     738ns  cuDeviceTotalMem
                    0.00%     684ns         2     342ns     176ns     508ns  cuDeviceGet
                    0.00%     423ns         1     423ns     423ns     423ns  cuModuleGetLoadingMode
                    0.00%      85ns         1      85ns      85ns      85ns  cuDeviceGetUuid
```

This CUDA file is a baseline for next days, as one might see that we have some warp divergence issues with L.50 ` if ((tid % (2 * s)) == 0)`.
In the first iteration of parallel reduction, only even threads execute the body of this conditional statement but all threads must be scheduled. On the second iteration, only one fourth of all threads are active but still all threads must be scheduled. 

Reference to book p.110.