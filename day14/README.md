# Day14

We saw in `Day14` that unrolling help with performance of the parallel reduction operator in CUDA. In `Day10` and `Day11` we also highlighted the presence of warp divergence in the last execution threads. If we consider the case when there are 32 or fewer threads left (that is, a single warp). Because warp execution is SIMT, there is implicit intra-warp synchronization after each instruction. The last 6 iterations of the reduction loop can therefore be unrolled. (cf book p.117). Since the book is rather old, now using **Warp-level Primitives** like `__shfl_down_sync` is recommended.


Profiling `parallelReduceUnroll.cu` : 

```bash
 nvprof ./parallelReduceUnroll2
==132799== NVPROF is profiling process 132799, command: ./parallelReduceUnroll2
Using CUDA Device: NVIDIA T600 Laptop GPU
Compute Capability: 7.5
Max Threads Per Block: 1024
Max Threads Per SM: 1024
Warp Size: 32
Total Global Memory: 3717.94 MiB
Unrooling 2 data blocks in the kernel.

--- Testing Parallel Reduction (Interleaved) ---

Problem Size N = 1 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -100
GPU Sum (actual):   -100
Verification: PASS
Elapsed Time: 0.014 ms
Performance:  0.000 GigaElements/s
Effective Bandwidth (kernel part): 0.001 GiB/s

Problem Size N = 128 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4672
GPU Sum (actual):   -4672
Verification: PASS
Elapsed Time: 0.007 ms
Performance:  0.017 GigaElements/s
Effective Bandwidth (kernel part): 0.065 GiB/s

Problem Size N = 256 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4160
GPU Sum (actual):   -4160
Verification: PASS
Elapsed Time: 0.008 ms
Performance:  0.031 GigaElements/s
Effective Bandwidth (kernel part): 0.116 GiB/s

Problem Size N = 257 elements
Configuration: 2 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.01 KiB
CPU Sum (expected): -4204
GPU Sum (actual):   -4204
Verification: PASS
Elapsed Time: 0.008 ms
Performance:  0.034 GigaElements/s
Effective Bandwidth (kernel part): 0.126 GiB/s

Problem Size N = 1048576 elements
Configuration: 4096 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -526400
GPU Sum (actual):   -526400
Verification: PASS
Elapsed Time: 0.154 ms
Performance:  6.787 GigaElements/s
Effective Bandwidth (kernel part): 25.383 GiB/s

Problem Size N = 1048589 elements
Configuration: 4097 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -525334
GPU Sum (actual):   -525334
Verification: PASS
Elapsed Time: 0.154 ms
Performance:  6.817 GigaElements/s
Effective Bandwidth (kernel part): 25.494 GiB/s

Problem Size N = 4194304 elements
Configuration: 16384 blocks, 256 threads/block
Input data: 16.00 MiB, Output (block sums): 64.00 KiB
CPU Sum (expected): -2102144
GPU Sum (actual):   -2102144
Verification: PASS
Elapsed Time: 0.569 ms
Performance:  7.371 GigaElements/s
Effective Bandwidth (kernel part): 27.567 GiB/s
-----------------------------------------------------

Done.
==132799== Profiling application: ./parallelReduceUnroll2
==132799== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.81%  7.3329ms        14  523.78us     512ns  2.9868ms  [CUDA memcpy HtoD]
                   18.84%  1.7101ms        14  122.15us  2.9760us  559.52us  reduceInterleavedUnroll(int*, int*, unsigned int)
                    0.21%  19.361us         7  2.7650us  1.3450us  6.9120us  [CUDA memcpy DtoH]
                    0.14%  12.353us         7  1.7640us     928ns  2.9440us  [CUDA memset]
      API calls:   54.87%  138.32ms         2  69.159ms     409ns  138.32ms  cudaEventCreate
                   38.17%  96.211ms         1  96.211ms  96.211ms  96.211ms  cudaDeviceReset
                    3.05%  7.6812ms        21  365.77us  2.0010us  3.2446ms  cudaMemcpy
                    1.29%  3.2450ms       114  28.464us     136ns  1.8331ms  cuDeviceGetAttribute
                    1.06%  2.6666ms         1  2.6666ms  2.6666ms  2.6666ms  cudaGetDeviceProperties
                    0.48%  1.2066ms         7  172.37us  4.6670us  635.77us  cudaEventSynchronize
                    0.43%  1.0908ms         7  155.84us  3.9500us  624.69us  cudaDeviceSynchronize
                    0.25%  631.07us        14  45.076us  1.2050us  161.80us  cudaMalloc
                    0.22%  553.42us        14  39.529us  1.6980us  135.79us  cudaFree
                    0.08%  200.72us        14  14.337us  2.0680us  113.28us  cudaLaunchKernel
                    0.04%  97.590us         1  97.590us  97.590us  97.590us  cudaGetDevice
                    0.03%  73.400us         7  10.485us  2.3370us  28.627us  cudaMemset
                    0.02%  53.892us        14  3.8490us  1.9160us  8.4920us  cudaEventRecord
                    0.01%  37.283us         1  37.283us  37.283us  37.283us  cuDeviceGetName
                    0.00%  10.312us         7  1.4730us     561ns  5.4020us  cudaEventElapsedTime
                    0.00%  7.3790us         1  7.3790us  7.3790us  7.3790us  cuDeviceTotalMem
                    0.00%  2.1780us         3     726ns     227ns  1.6640us  cuDeviceGetCount
                    0.00%  1.9660us         2     983ns     288ns  1.6780us  cudaEventDestroy
                    0.00%  1.6650us        14     118ns      39ns     418ns  cudaGetLastError
                    0.00%     970ns         1     970ns     970ns     970ns  cuDeviceGetPCIBusId
                    0.00%     735ns         2     367ns     162ns     573ns  cuDeviceGet
                    0.00%     422ns         1     422ns     422ns     422ns  cuModuleGetLoadingMode
                    0.00%     274ns         1     274ns     274ns     274ns  cuDeviceGetUuid
```

Compared to `Day13` we clearly see the increase of performance due to CUDA primitives that "allows the data exchange is performed between registers, and more efficient than going through shared memory, which requires a load, a store and an extra register to hold the address."