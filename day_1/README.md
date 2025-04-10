# Day 1 

CUDA vector add operation. 

```bash
nvprof ./vectorAdd
Starting Simple Vector Add (Single Stream) - Timing Total GPU Work...
Configuration:
Vector elements (N): 1048576 (1.00 Million)
Vector size: 4194304 bytes (4.00 MB)
Kernel launch: 4096 blocks, 256 threads per block
Allocating host memory (16777216 bytes)...
Initializing host data...
Host data initialization time: 0.03250 sec
Performing vector addition on host...
Host vector addition time: 0.00299 sec
Allocating Device Memory (12582912 bytes)...
==754875== NVPROF is profiling process 754875, command: ./vectorAdd

Creating CUDA events for overall GPU timing...
Starting GPU operations and recording events...
  Copying data from host to device (A)...
  Copying data from host to device (B)...
  Launching kernel...
  Copying result from device to host...
Waiting for all GPU operations to complete (synchronizing on event)...

--- Total GPU Performance ---
Total GPU execution time (H2D + Kernel + D2H): 2.536 ms
Effective Memory Bandwidth (Combined H2D + Kernel + D2H): 4.621 GiB/s
Overall Performance including transfers: 0.413 GFLOPS

--- Verification ---
Arrays match.

Cleaning up resources...
Done
==754875== Profiling application: ./vectorAdd
==754875== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.59%  1.2086ms         2  604.29us  576.71us  631.87us  [CUDA memcpy HtoD]
                   29.25%  593.15us         1  593.15us  593.15us  593.15us  [CUDA memcpy DtoH]
                   11.16%  226.27us         1  226.27us  226.27us  226.27us  vectorAddOnDevice(float*, float*, float*, int)
      API calls:   94.84%  139.81ms         3  46.602ms  37.851us  139.71ms  cudaMalloc
                    2.69%  3.9687ms       114  34.813us     164ns  2.1162ms  cuDeviceGetAttribute
                    1.62%  2.3902ms         3  796.73us  700.15us  973.88us  cudaMemcpy
                    0.70%  1.0264ms         3  342.14us  38.664us  948.95us  cudaFree
                    0.08%  122.62us         1  122.62us  122.62us  122.62us  cudaLaunchKernel
                    0.03%  47.373us         1  47.373us  47.373us  47.373us  cuDeviceGetName
                    0.02%  30.455us         2  15.227us  5.2110us  25.244us  cudaEventRecord
                    0.01%  9.6800us         2  4.8400us  1.3550us  8.3250us  cudaEventCreate
                    0.00%  5.2380us         1  5.2380us  5.2380us  5.2380us  cuDeviceGetPCIBusId
                    0.00%  3.3560us         3  1.1180us     228ns  2.7010us  cuDeviceGetCount
                    0.00%  1.7890us         1  1.7890us  1.7890us  1.7890us  cudaEventSynchronize
                    0.00%  1.6010us         1  1.6010us  1.6010us  1.6010us  cudaEventElapsedTime
                    0.00%  1.1090us         2     554ns     335ns     774ns  cudaEventDestroy
                    0.00%     929ns         2     464ns     188ns     741ns  cuDeviceGet
                    0.00%     696ns         1     696ns     696ns     696ns  cuDeviceTotalMem
                    0.00%     472ns         1     472ns     472ns     472ns  cuModuleGetLoadingMode
                    0.00%     328ns         1     328ns     328ns     328ns  cuDeviceGetUuid
                    0.00%     324ns         1     324ns     324ns     324ns  cudaGetLastError
```

Here is illustration of what happening under the hood :
```
# Vector Addition: C[i] = A[i] + B[i]
# Vector Size N = 16
# Threads Per Block (TPB / blockDim.x) = 4
# Number of Blocks = 4 (Block 0, Block 1, Block 2, Block 3)

# --- Input/Output Vectors ---
# (Showing indices relevant to the calculation)

Vector Indices (i):    0    1    2    3      4    5    6    7      8    9   10   11     12   13   14   15
Vector A:            | A0 | A1 | A2 | A3 |  | A4 | A5 | A6 | A7 |  | A8 | A9 |A10 |A11 |  |A12 |A13 |A14 |A15 |
Vector B:            | B0 | B1 | B2 | B3 |  | B4 | B5 | B6 | B7 |  | B8 | B9 |B10 |B11 |  |B12 |B13 |B14 |B15 |
                     +------------------+  +------------------+  +------------------+  +------------------+
Vector C (Result):   | C0 | C1 | C2 | C3 |  | C4 | C5 | C6 | C7 |  | C8 | C9 |C10 |C11 |  |C12 |C13 |C14 |C15 |
                     +==================+  +==================+  +==================+  +==================+
                           ^                   ^                   ^                   ^
                           |                   |                   |                   |
# --- CUDA Grid Structure & Mapping ---        |                   |                   |
                           |                   |                   |                   |
                     +-----+-------------------+-------------------+-------------------+
                     |                         |                   |                   |
Grid:           [ Block 0 ]--------- [ Block 1 ]--------- [ Block 2 ]--------- [ Block 3 ]
blockIdx.x:        (bIdx=0)            (bIdx=1)            (bIdx=2)            (bIdx=3)
                   / | \ \             / | \ \             / | \ \             / | \ \
Threads:         T0 T1 T2 T3         T0 T1 T2 T3         T0 T1 T2 T3         T0 T1 T2 T3
threadIdx.x:    (tIdx=0..3)         (tIdx=0..3)         (tIdx=0..3)         (tIdx=0..3)
blockDim.x:        (4)                 (4)                 (4)                 (4)

# --- How Each Thread Calculates its Global Index (idx) ---
# Formula: idx = blockIdx.x * blockDim.x + threadIdx.x

# Block 0 (bIdx=0):
# T0 (tIdx=0): idx = 0 * 4 + 0 = 0   -> Calculates C[0] = A[0] + B[0]
# T1 (tIdx=1): idx = 0 * 4 + 1 = 1   -> Calculates C[1] = A[1] + B[1]
# T2 (tIdx=2): idx = 0 * 4 + 2 = 2   -> Calculates C[2] = A[2] + B[2]
# T3 (tIdx=3): idx = 0 * 4 + 3 = 3   -> Calculates C[3] = A[3] + B[3]

# Block 1 (bIdx=1):
# T0 (tIdx=0): idx = 1 * 4 + 0 = 4   -> Calculates C[4] = A[4] + B[4]
# T1 (tIdx=1): idx = 1 * 4 + 1 = 5   -> Calculates C[5] = A[5] + B[5]
# T2 (tIdx=2): idx = 1 * 4 + 2 = 6   -> Calculates C[6] = A[6] + B[6]
# T3 (tIdx=3): idx = 1 * 4 + 3 = 7   -> Calculates C[7] = A[7] + B[7]

# Block 2 (bIdx=2):
# T0 (tIdx=0): idx = 2 * 4 + 0 = 8   -> Calculates C[8] = A[8] + B[8]
# T1 (tIdx=1): idx = 2 * 4 + 1 = 9   -> Calculates C[9] = A[9] + B[9]
# T2 (tIdx=2): idx = 2 * 4 + 2 = 10  -> Calculates C[10]= A[10]+ B[10]
# T3 (tIdx=3): idx = 2 * 4 + 3 = 11  -> Calculates C[11]= A[11]+ B[11]

# Block 3 (bIdx=3):
# T0 (tIdx=0): idx = 3 * 4 + 0 = 12  -> Calculates C[12]= A[12]+ B[12]
# T1 (tIdx=1): idx = 3 * 4 + 1 = 13  -> Calculates C[13]= A[13]+ B[13]
# T2 (tIdx=2): idx = 3 * 4 + 2 = 14  -> Calculates C[14]= A[14]+ B[14]
# T3 (tIdx=3): idx = 3 * 4 + 3 = 15  -> Calculates C[15]= A[15]+ B[15]

# --- Summary ---
# - Each Block (identified by blockIdx.x) handles a contiguous chunk of the output vector.
# - Each Thread within a block (identified by threadIdx.x) handles one specific element within that block's chunk.
# - blockDim.x tells us how many threads are in each block (the size of the chunk a block processes).
# - The global index formula combines blockIdx.x, blockDim.x, and threadIdx.x to give each thread a unique index 'i'
#   into the vectors A, B, and C that it is responsible for.
```

Roughly 65% for allocating memory and 30% reset all state's device and only 5% of the execution is used for vector operation. 

Quote from book p.48 : 

*If your application spends more time computing than transferring data, then it may be
possible to overlap these operations and completely hide the latency associated with transferring
data. If your application spends less time computing than transferring data, it is important to
minimize the transfer between the host and device. In Chapter 6, you will learn how to overlap
computation with communication using CUDA streams and events.*

We will see in day 2 how CUDA streams can help us shed some time in data transfers.