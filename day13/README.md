# Day13

Adding `g_idata[block_start_idx + tid] += g_idata[block_start_idx + tid + blockDim.x]` allows use to have each thread add an element from the neighboring data block. Now you only need half as many threads to process the same data set. (cf book p.116)

Profiling `parallelReduceUnroll.cu` : 
```bash
nvprof ./parallelReduceUnroll
==42086== NVPROF is profiling process 42086, command: ./parallelReduceUnroll
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
Elapsed Time: 0.015 ms
Performance:  0.000 GigaElements/s
Effective Bandwidth (kernel part): 0.000 GiB/s

Problem Size N = 128 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4672
GPU Sum (actual):   -4672
Verification: PASS
Elapsed Time: 0.010 ms
Performance:  0.013 GigaElements/s
Effective Bandwidth (kernel part): 0.050 GiB/s

Problem Size N = 256 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4160
GPU Sum (actual):   -4160
Verification: PASS
Elapsed Time: 0.010 ms
Performance:  0.026 GigaElements/s
Effective Bandwidth (kernel part): 0.096 GiB/s

Problem Size N = 257 elements
Configuration: 2 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.01 KiB
CPU Sum (expected): -4204
GPU Sum (actual):   -4204
Verification: PASS
Elapsed Time: 0.009 ms
Performance:  0.028 GigaElements/s
Effective Bandwidth (kernel part): 0.107 GiB/s

Problem Size N = 1048576 elements
Configuration: 4096 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -526400
GPU Sum (actual):   -526400
Verification: PASS
Elapsed Time: 0.266 ms
Performance:  3.948 GigaElements/s
Effective Bandwidth (kernel part): 14.765 GiB/s

Problem Size N = 1048589 elements
Configuration: 4097 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -525334
GPU Sum (actual):   -525334
Verification: PASS
Elapsed Time: 0.268 ms
Performance:  3.910 GigaElements/s
Effective Bandwidth (kernel part): 14.624 GiB/s

Problem Size N = 4194304 elements
Configuration: 16384 blocks, 256 threads/block
Input data: 16.00 MiB, Output (block sums): 64.00 KiB
CPU Sum (expected): -2102144
GPU Sum (actual):   -2102144
Verification: PASS
Elapsed Time: 1.031 ms
Performance:  4.067 GigaElements/s
Effective Bandwidth (kernel part): 15.210 GiB/s
-----------------------------------------------------
Unrooling 4 data blocks in the kernel.

--- Testing Parallel Reduction (Interleaved) ---

Problem Size N = 1 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -100
GPU Sum (actual):   -100
Verification: PASS
Elapsed Time: 0.011 ms
Performance:  0.000 GigaElements/s
Effective Bandwidth (kernel part): 0.001 GiB/s

Problem Size N = 128 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4672
GPU Sum (actual):   -4672
Verification: PASS
Elapsed Time: 0.008 ms
Performance:  0.015 GigaElements/s
Effective Bandwidth (kernel part): 0.057 GiB/s

Problem Size N = 256 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4160
GPU Sum (actual):   -4160
Verification: PASS
Elapsed Time: 0.009 ms
Performance:  0.030 GigaElements/s
Effective Bandwidth (kernel part): 0.111 GiB/s

Problem Size N = 257 elements
Configuration: 2 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.01 KiB
CPU Sum (expected): -4204
GPU Sum (actual):   -4204
Verification: PASS
Elapsed Time: 0.009 ms
Performance:  0.029 GigaElements/s
Effective Bandwidth (kernel part): 0.108 GiB/s

Problem Size N = 1048576 elements
Configuration: 4096 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -526400
GPU Sum (actual):   -526400
Verification: PASS
Elapsed Time: 0.240 ms
Performance:  4.373 GigaElements/s
Effective Bandwidth (kernel part): 16.355 GiB/s

Problem Size N = 1048589 elements
Configuration: 4097 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -525334
GPU Sum (actual):   -525334
Verification: PASS
Elapsed Time: 0.239 ms
Performance:  4.387 GigaElements/s
Effective Bandwidth (kernel part): 16.405 GiB/s

Problem Size N = 4194304 elements
Configuration: 16384 blocks, 256 threads/block
Input data: 16.00 MiB, Output (block sums): 64.00 KiB
CPU Sum (expected): -2102144
GPU Sum (actual):   -2102144
Verification: PASS
Elapsed Time: 0.928 ms
Performance:  4.518 GigaElements/s
Effective Bandwidth (kernel part): 16.898 GiB/s
-----------------------------------------------------

Done.
==42086== Profiling application: ./parallelReduceUnroll
==42086== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   71.64%  15.138ms        28  540.63us     448ns  3.5063ms  [CUDA memcpy HtoD]
                   14.75%  3.1178ms        14  222.70us  4.4160us  1.0223ms  reduceInterleavedUnroll(int*, int*, unsigned int)
                   13.32%  2.8142ms        14  201.01us  3.9360us  922.08us  reduceInterleavedUnroll2(int*, int*, unsigned int)
                    0.18%  38.079us        14  2.7190us  1.2480us  7.0400us  [CUDA memcpy DtoH]
                    0.11%  22.847us        14  1.6310us     832ns  2.8800us  [CUDA memset]
      API calls:   67.21%  262.96ms         4  65.739ms     317ns  140.36ms  cudaEventCreate
                   24.56%  96.074ms         1  96.074ms  96.074ms  96.074ms  cudaDeviceReset
                    4.08%  15.950ms        42  379.77us  1.9680us  3.6531ms  cudaMemcpy
                    0.91%  3.5649ms        14  254.64us  5.3990us  1.1018ms  cudaEventSynchronize
                    0.88%  3.4522ms        14  246.58us  5.1520us  1.0765ms  cudaDeviceSynchronize
                    0.84%  3.2899ms       114  28.858us     159ns  1.8755ms  cuDeviceGetAttribute
                    0.69%  2.6885ms         1  2.6885ms  2.6885ms  2.6885ms  cudaGetDeviceProperties
                    0.32%  1.2552ms        28  44.828us  1.3930us  154.02us  cudaMalloc
                    0.29%  1.1275ms        28  40.268us  1.7920us  108.60us  cudaFree
                    0.11%  421.00us        28  15.035us  2.1160us  120.19us  cudaLaunchKernel
                    0.04%  175.79us        14  12.556us  2.5420us  42.140us  cudaMemset
                    0.03%  99.068us        28  3.5380us  1.9350us  10.686us  cudaEventRecord
                    0.02%  87.689us         1  87.689us  87.689us  87.689us  cudaGetDevice
                    0.01%  43.275us         1  43.275us  43.275us  43.275us  cuDeviceGetName
                    0.00%  11.369us        14     812ns     522ns  1.2400us  cudaEventElapsedTime
                    0.00%  8.0480us         1  8.0480us  8.0480us  8.0480us  cuDeviceTotalMem
                    0.00%  3.6230us        28     129ns      45ns     437ns  cudaGetLastError
                    0.00%  2.9420us         4     735ns     222ns  1.5170us  cudaEventDestroy
                    0.00%  2.3920us         3     797ns     219ns  1.8880us  cuDeviceGetCount
                    0.00%  1.0990us         1  1.0990us  1.0990us  1.0990us  cuDeviceGetPCIBusId
                    0.00%     960ns         2     480ns     156ns     804ns  cuDeviceGet
                    0.00%     499ns         1     499ns     499ns     499ns  cuModuleGetLoadingMode
                    0.00%     469ns         1     469ns     469ns     469ns  cuDeviceGetUuid
```

We also added a function that unroll 4 data blocks, we also see some performance improvement. Just as you might expect, more independent memory load/store operations in a single thread yield better performance as memory latency can be better hidden.