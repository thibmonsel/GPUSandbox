# Day 1 

CUDA vector add operation. 

```bash
./vectorAdd
Starting Simple Vector Add (Single Stream) - Timing Total GPU Work...
Configuration:
Vector elements (N): 1048576 (1.00 Million)
Vector size: 4194304 bytes (4.00 MB)
Kernel launch: 4096 blocks, 256 threads per block
Allocating host memory (16777216 bytes)...
Initializing host data...
Host data initialization time: 0.03303 sec
Performing vector addition on host...
Host vector addition time: 0.00279 sec
Allocating Device Memory (12582912 bytes)...

Creating CUDA events for overall GPU timing...
Starting GPU operations and recording events...
  Copying data from host to device (A)...
  Copying data from host to device (B)...
  Launching kernel...
  Copying result from device to host...
Waiting for all GPU operations to complete (synchronizing on event)...

--- Total GPU Performance ---
Total GPU execution time (H2D + Kernel + D2H): 2.266 ms
Effective Memory Bandwidth (Combined H2D + Kernel + D2H): 5.172 GiB/s
Overall Performance including transfers: 0.463 GFLOPS

--- Verification ---
Arrays match.

Cleaning up resources...
Done
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

Profiling of the code with `nvprof`.

```bash
==477106== Profiling application: ./vectorAdd
==477106== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.34%  1.2360ms         2  617.98us  601.47us  634.50us  [CUDA memcpy HtoD]
                   30.77%  591.10us         1  591.10us  591.10us  591.10us  [CUDA memcpy DtoH]
                    4.89%  93.984us         1  93.984us  93.984us  93.984us  vectorAddOnDevice(float*, float*, float*, int)
      API calls:   95.26%  135.98ms         3  45.326ms  36.698us  135.90ms  cudaMalloc
                    2.95%  4.2039ms       114  36.876us     168ns  2.3437ms  cuDeviceGetAttribute
                    1.57%  2.2360ms         3  745.33us  700.77us  805.26us  cudaMemcpy
                    0.10%  135.97us         3  45.322us  32.421us  68.650us  cudaFree
                    0.08%  110.95us         1  110.95us  110.95us  110.95us  cudaLaunchKernel
                    0.03%  36.595us         1  36.595us  36.595us  36.595us  cuDeviceGetName
                    0.01%  15.733us         2  7.8660us  5.4860us  10.247us  cudaEventRecord
                    0.01%  9.5410us         2  4.7700us  1.2050us  8.3360us  cudaEventCreate
                    0.00%  4.7160us         1  4.7160us  4.7160us  4.7160us  cuDeviceGetPCIBusId
                    0.00%  3.0990us         3  1.0330us     215ns  2.6100us  cuDeviceGetCount
                    0.00%  1.8460us         1  1.8460us  1.8460us  1.8460us  cudaEventSynchronize
                    0.00%  1.3570us         1  1.3570us  1.3570us  1.3570us  cudaEventElapsedTime
                    0.00%     981ns         2     490ns     169ns     812ns  cuDeviceGet
                    0.00%     899ns         2     449ns     246ns     653ns  cudaEventDestroy
                    0.00%     874ns         1     874ns     874ns     874ns  cuDeviceTotalMem
                    0.00%     394ns         1     394ns     394ns     394ns  cuModuleGetLoadingMode
                    0.00%     288ns         1     288ns     288ns     288ns  cuDeviceGetUuid
                    0.00%     177ns         1     177ns     177ns     177ns  cudaGetLastError
```

Roughly 60% for allocating memory and 40% reset all state's device and only 10% of the execution is used for vector operation. 

Quote from book p.48 : 

*If your application spends more time computing than transferring data, then it may be
possible to overlap these operations and completely hide the latency associated with transferring
data. If your application spends less time computing than transferring data, it is important to
minimize the transfer between the host and device. In Chapter 6, you will learn how to overlap
computation with communication using CUDA streams and events.*

We will see in day 2 how CUDA streams can help us shed some time in data transfers.