# Day13

Visual example of a [Parallel reduction](https://en.wikipedia.org/wiki/Reduction_operator) with an Interleaved pair method: The interleaved pair approach reverses the striding of elements compared to the neighbored approach: The stride is started at half of the thread block size and then reduced by half on each iteration. Each thread adds two elements separated by the current stride to produce a partial sum at each round. However, the load/store locations in global memory for each
thread are different (p.112 from book)

```bash
Initial Array (N=8):
[ 3, 1, 7, 0, 4, 1, 6, 3 ]
--------------------------------------------------

### STEP 1 ###
- Stride: 4
- Active Threads: tid = 0, 1, 2, 3
- Action: data[tid] += data[tid + 4]

  - Thread 0: data[0] = data[0] + data[4]  =>  3 + 4 = 7
  - Thread 1: data[1] = data[1] + data[5]  =>  1 + 1 = 2
  - Thread 2: data[2] = data[2] + data[6]  =>  7 + 6 = 13
  - Thread 3: data[3] = data[3] + data[7]  =>  0 + 3 = 3

- Array State after Step 1:
[ 7, 2, 13, 3, 4, 1, 6, 3 ]
--------------------------------------------------

### STEP 2 ###
- Stride: 2
- Active Threads: tid = 0, 1
- Action: data[tid] += data[tid + 2]

  - Thread 0: data[0] = data[0] + data[2]  =>  7 + 13 = 20
  - Thread 1: data[1] = data[1] + data[3]  =>  2 + 3  = 5

- Array State after Step 2:
[ 20, 5, 13, 3, 4, 1, 6, 3 ]
--------------------------------------------------

### STEP 3 ###
- Stride: 1
- Active Threads: tid = 0
- Action: data[tid] += data[tid + 1]

  - Thread 0: data[0] = data[0] + data[1]  =>  20 + 5 = 25

- Array State after Step 3:
[ 25, 5, 13, 3, 4, 1, 6, 3 ]
```

Profiling `parallelReduce3.cu` : 

```bash 
command: ./reduceParallel3
Using CUDA Device: NVIDIA T600 Laptop GPU
Compute Capability: 7.5
Max Threads Per Block: 1024
Max Threads Per SM: 1024
Warp Size: 32
Total Global Memory: 3717.94 MiB

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
Elapsed Time: 0.012 ms
Performance:  0.011 GigaElements/s
Effective Bandwidth (kernel part): 0.040 GiB/s

Problem Size N = 256 elements
Configuration: 1 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.00 KiB
CPU Sum (expected): -4160
GPU Sum (actual):   -4160
Verification: PASS
Elapsed Time: 0.013 ms
Performance:  0.020 GigaElements/s
Effective Bandwidth (kernel part): 0.074 GiB/s

Problem Size N = 257 elements
Configuration: 2 blocks, 256 threads/block
Input data: 0.00 MiB, Output (block sums): 0.01 KiB
CPU Sum (expected): -4204
GPU Sum (actual):   -4204
Verification: PASS
Elapsed Time: 0.009 ms
Performance:  0.029 GigaElements/s
Effective Bandwidth (kernel part): 0.110 GiB/s

Problem Size N = 1048576 elements
Configuration: 4096 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -526400
GPU Sum (actual):   -526400
Verification: PASS
Elapsed Time: 0.293 ms
Performance:  3.582 GigaElements/s
Effective Bandwidth (kernel part): 13.398 GiB/s

Problem Size N = 1048589 elements
Configuration: 4097 blocks, 256 threads/block
Input data: 4.00 MiB, Output (block sums): 16.00 KiB
CPU Sum (expected): -525334
GPU Sum (actual):   -525334
Verification: PASS
Elapsed Time: 0.299 ms
Performance:  3.506 GigaElements/s
Effective Bandwidth (kernel part): 13.112 GiB/s

Problem Size N = 4194304 elements
Configuration: 16384 blocks, 256 threads/block
Input data: 16.00 MiB, Output (block sums): 64.00 KiB
CPU Sum (expected): -2102144
GPU Sum (actual):   -2102144
Verification: PASS
Elapsed Time: 1.154 ms
Performance:  3.634 GigaElements/s
Effective Bandwidth (kernel part): 13.592 GiB/s
-----------------------------------------------------

Done.
==5475== Profiling application: ./reduceParallel3
==5475== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   70.11%  8.1812ms        14  584.37us     512ns  3.6184ms  [CUDA memcpy HtoD]
                   29.63%  3.4573ms        14  246.95us  4.1920us  1.1407ms  reduceInterleaved(int*, int*, unsigned int)
                    0.16%  18.656us         7  2.6650us  1.3440us  7.0080us  [CUDA memcpy DtoH]
                    0.11%  12.256us         7  1.7500us     960ns  2.9440us  [CUDA memset]
      API calls:   53.69%  171.30ms         2  85.649ms     444ns  171.30ms  cudaEventCreate
                   29.90%  95.389ms         1  95.389ms  95.389ms  95.389ms  cudaDeviceReset
                    9.74%  31.069ms        14  2.2192ms  2.3240us  30.909ms  cudaLaunchKernel
                    2.74%  8.7503ms        21  416.68us  2.0470us  3.7229ms  cudaMemcpy
                    1.01%  3.2167ms       114  28.216us     151ns  1.9478ms  cuDeviceGetAttribute
                    0.84%  2.6688ms         1  2.6688ms  2.6688ms  2.6688ms  cudaGetDeviceProperties
                    0.64%  2.0292ms         7  289.89us  5.4130us  1.2180ms  cudaEventSynchronize
                    0.63%  2.0034ms         7  286.20us  3.6750us  1.1859ms  cudaDeviceSynchronize
                    0.28%  905.30us        14  64.664us  1.9770us  319.63us  cudaMalloc
                    0.26%  838.50us        14  59.892us  3.3560us  165.82us  cudaFree
                    0.12%  396.55us         1  396.55us  396.55us  396.55us  cuDeviceTotalMem
                    0.08%  245.17us         7  35.024us  4.5410us  156.14us  cudaMemset
                    0.04%  114.32us         1  114.32us  114.32us  114.32us  cudaGetDevice
                    0.02%  60.165us        14  4.2970us  2.0190us  9.5510us  cudaEventRecord
                    0.01%  31.663us         1  31.663us  31.663us  31.663us  cuDeviceGetName
                    0.00%  8.5290us         7  1.2180us     595ns  1.7310us  cudaEventElapsedTime
                    0.00%  3.9160us         1  3.9160us  3.9160us  3.9160us  cuDeviceGetPCIBusId
                    0.00%  3.7500us        14     267ns      45ns  1.5520us  cudaGetLastError
                    0.00%  2.9330us         3     977ns     198ns  2.4700us  cuDeviceGetCount
                    0.00%  1.8410us         2     920ns     348ns  1.4930us  cudaEventDestroy
                    0.00%     842ns         2     421ns     159ns     683ns  cuDeviceGet
                    0.00%     527ns         1     527ns     527ns     527ns  cuModuleGetLoadingMode
                    0.00%     291ns         1     291ns     291ns     291ns  cuDeviceGetUuid
```


`reduceInterleaved` (day13) also maintains the same amount of warp divergence as `reduceNeighboredLess` (day12). This performance improvement is primarily a result of the global memory load and store patterns in `reduceInterleaved`. To achieve high bandwidth, shared memory is not a single monolithic block. It is divided into 32 equally-sized memory modules called banks. There are 32 banks. Successive 32-bit words are assigned to successive banks. So, `sdata[0]` is in bank 0, `sdata[1]` is in bank 1, ..., `sdata[31]` is in bank 31, and `sdata[32]` is back in bank 0. The bank for an address `addr` is determined by `addr % 32`. A bank can service only one read or one write request per cycle from a single warp.

Explanation :

Let's assume our block has 512 threads, and we are analyzing the first warp (threads 0-31).

### Step 1 (stride = 256):

The if (tid < 256) condition is true for all threads in our warp (0-31). The warp is 100% active. Each thread t reads from `sdata[t + 256]`.

- Thread 0 reads `sdata[256]`. Bank is 256 % 32 = 0.
- Thread 1 reads `sdata[257]`. Bank is 257 % 32 = 1.
- Thread 2 reads `sdata[258]`. Bank is 258 % 32 = 2.
- ...
- Thread 31 reads `sdata[287]`. Bank is 287 % 32 = 31.

Result: All 32 threads in the warp access a different bank (0 through 31). This is a perfect, conflict-free, coalesced access. The entire read operation for the warp completes in a single cycle.

### Step 2 (stride = 128): 

Same logic. All 32 threads access `sdata[t + 128]`, which maps to banks 0 through 31. Conflict-free.

### Step ... (stride = 32): 

All 32 threads access `sdata[t + 32]`.

- Thread 0 reads `sdata[32]`. Bank is 32 % 32 = 0.
- Thread 1 reads `sdata[33]`. Bank is 33 % 32 = 1.
- ...
- Thread 31 reads `sdata[63]`. Bank is 63 % 32 = 31.
    
Result: Still a perfect, conflict-free access.