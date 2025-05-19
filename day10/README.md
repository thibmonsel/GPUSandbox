# Day10

Visual example of a [Parallel reduction](https://en.wikipedia.org/wiki/Reduction_operator) with a Neighbored pair method: Elements are paired with their immediate neighbor. In this implementation, a thread takes two adjacent elements to produce one partial sum at each step.  For an array with N elements, this implementation requires $N − 1$ sums and $\log_2(N)$ steps.

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


Profiling `vectorReduce.cu` : 

```bash
nvprof ./vectorReduce
==262099== NVPROF is profiling process 262099, command: ./vectorReduce
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
Elapsed Time: 0.016 ms
Performance:  0.000 GigaElements/s
Effective Bandwidth (kernel part): 0.000 GiB/s

Problem Size N = 128 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4672
GPU Sum (actual):   -4672
Verification: PASS
Elapsed Time: 0.014 ms
Performance:  0.009 GigaElements/s
Effective Bandwidth (kernel part): 0.035 GiB/s

Problem Size N = 256 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4160
GPU Sum (actual):   -4160
Verification: PASS
Elapsed Time: 0.012 ms
Performance:  0.021 GigaElements/s
Effective Bandwidth (kernel part): 0.079 GiB/s

Problem Size N = 257 elements
Configuration: 2 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.01 KiB
CPU Sum (expected): -4204
GPU Sum (actual):   -4204
Verification: PASS
Elapsed Time: 0.013 ms
Performance:  0.020 GigaElements/s
Effective Bandwidth (kernel part): 0.075 GiB/s

Problem Size N = 1048576 elements
Configuration: 4096 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -526400
GPU Sum (actual):   -526400
Verification: PASS
Elapsed Time: 0.359 ms
Performance:  2.918 GigaElements/s
Effective Bandwidth (kernel part): 10.914 GiB/s

Problem Size N = 1048589 elements
Configuration: 4097 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -525334
GPU Sum (actual):   -525334
Verification: PASS
Elapsed Time: 0.356 ms
Performance:  2.944 GigaElements/s
Effective Bandwidth (kernel part): 11.009 GiB/s

Problem Size N = 4194304 elements
Configuration: 16384 blocks, 256 threads/block
Input data: 16.00 MiB, Output (block sums): 64.00 KiB
CPU Sum (expected): -2102144
GPU Sum (actual):   -2102144
Verification: PASS
Elapsed Time: 1.391 ms
Performance:  3.014 GigaElements/s
Effective Bandwidth (kernel part): 11.273 GiB/s
-----------------------------------------------------

Done.
==262099== Profiling application: ./vectorReduce
==262099== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.08%  8.6348ms        14  616.77us     512ns  3.1952ms  [CUDA memcpy HtoD]
                   32.67%  4.2060ms        14  300.43us  4.5440us  1.3851ms  reduceNeighbored(int*, int*, unsigned int)
                    0.15%  19.487us         7  2.7830us  1.4080us  7.7440us  [CUDA memcpy DtoH]
                    0.10%  12.640us         7  1.8050us     960ns  3.0720us  [CUDA memset]
      API calls:   53.00%  148.20ms         2  74.101ms     513ns  148.20ms  cudaEventCreate
                   35.85%  100.24ms         1  100.24ms  100.24ms  100.24ms  cudaDeviceReset
                    3.28%  9.1786ms        21  437.07us  3.4120us  3.2790ms  cudaMemcpy
                    2.75%  7.6815ms        14  548.68us  4.0730us  7.5088ms  cudaLaunchKernel
                    1.39%  3.8725ms       114  33.968us      47ns  2.2158ms  cuDeviceGetAttribute
                    1.16%  3.2561ms         1  3.2561ms  3.2561ms  3.2561ms  cudaGetDeviceProperties
                    0.85%  2.3744ms         7  339.20us  3.4460us  1.4419ms  cudaEventSynchronize
                    0.82%  2.2996ms         7  328.51us  4.8210us  1.4419ms  cudaDeviceSynchronize
                    0.43%  1.2028ms        14  85.913us  3.4740us  345.71us  cudaFree
                    0.38%  1.0600ms        14  75.717us  2.4110us  268.57us  cudaMalloc
                    0.03%  96.012us         7  13.716us  4.8820us  36.801us  cudaMemset
                    0.03%  83.096us        14  5.9350us  2.6030us  15.181us  cudaEventRecord
                    0.01%  18.873us         1  18.873us  18.873us  18.873us  cuDeviceGetName
                    0.01%  15.735us         7  2.2470us  1.1020us  7.4200us  cudaEventElapsedTime
                    0.00%  4.4230us         1  4.4230us  4.4230us  4.4230us  cuDeviceGetPCIBusId
                    0.00%  2.8940us         3     964ns      87ns  2.5630us  cuDeviceGetCount
                    0.00%  2.6340us         2  1.3170us     323ns  2.3110us  cudaEventDestroy
                    0.00%  2.5360us        14     181ns      83ns     386ns  cudaGetLastError
                    0.00%  1.6350us         1  1.6350us  1.6350us  1.6350us  cudaGetDevice
                    0.00%     801ns         2     400ns      48ns     753ns  cuDeviceGet
                    0.00%     646ns         1     646ns     646ns     646ns  cuModuleGetLoadingMode
                    0.00%     231ns         1     231ns     231ns     231ns  cuDeviceTotalMem
                    0.00%      93ns         1      93ns      93ns      93ns  cuDeviceGetUuid
```

This CUDA file is a baseline for next days, as one might see that we have some warp divergence issues with L.50 ` if ((tid % (2 * s)) == 0)`.
In the first iteration of parallel reduction, only even threads execute the body of this conditional statement but all threads must be scheduled. On the second iteration, only one fourth of all threads are active but still all threads must be scheduled. 

Reference to book p.110.