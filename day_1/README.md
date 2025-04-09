# Day 1 

CUDA vector add operation. 

```bash
Starting... 
Vector size: 1048576 that will be of size 4194304 bytes (sizeof(float)=4) 
Init time for vectors: 0.04557 sec
Vector addition on the host: 0.00231 sec
Vector addition on the device: 0.00035 sec
Arrays match.
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
nvprof ./vectorAdd
Starting... 
Vector size: 1048576 that will be of size 4194304 bytes (sizeof(float)=4) 
Init time for vectors: 0.02718 sec
Vector addition on the host: 0.00217 sec
==283523== NVPROF is profiling process 283523, command: ./vectorAdd
Vector addition on the device: 0.00034 sec
Arrays match.
Done
==283523== Profiling application: ./vectorAdd
==283523== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.31%  1.3775ms         2  688.76us  639.36us  738.17us  [CUDA memcpy HtoD]
                   35.00%  856.28us         1  856.28us  856.28us  856.28us  [CUDA memcpy DtoH]
                    8.69%  212.48us         1  212.48us  212.48us  212.48us  vectorAddOnDevice(float*, float*, float*, int)
      API calls:   59.21%  147.90ms         3  49.302ms  35.819us  147.82ms  cudaMalloc
                   37.77%  94.343ms         1  94.343ms  94.343ms  94.343ms  cudaDeviceReset
                    1.75%  4.3713ms       114  38.345us     124ns  2.4519ms  cuDeviceGetAttribute
                    1.06%  2.6406ms         3  880.21us  769.17us  996.28us  cudaMemcpy
                    0.09%  213.63us         1  213.63us  213.63us  213.63us  cudaDeviceSynchronize
                    0.06%  162.24us         3  54.081us  32.844us  75.560us  cudaFree
                    0.05%  118.45us         1  118.45us  118.45us  118.45us  cudaLaunchKernel
                    0.02%  39.638us         1  39.638us  39.638us  39.638us  cuDeviceGetName
                    0.00%  4.6680us         1  4.6680us  4.6680us  4.6680us  cuDeviceGetPCIBusId
                    0.00%  3.1710us         3  1.0570us     234ns  2.2870us  cuDeviceGetCount
                    0.00%  1.4800us         2     740ns     148ns  1.3320us  cuDeviceGet
                    0.00%     799ns         1     799ns     799ns     799ns  cuDeviceTotalMem
                    0.00%     328ns         1     328ns     328ns     328ns  cuModuleGetLoadingMode
                    0.00%     239ns         1     239ns     239ns     239ns  cuDeviceGetUuid
```

Roughly 60% for allocating memory and 40% reset all state's device and only 10% of the execution is used for vector operation. 

Quote from book p.48 : 

*If your application spends more time computing than transferring data, then it may be
possible to overlap these operations and completely hide the latency associated with transferring
data. If your application spends less time computing than transferring data, it is important to
minimize the transfer between the host and device. In Chapter 6, you will learn how to overlap
computation with communication using CUDA streams and events.*

In our case here the data transfer can't be sped up because the size of our vector is small and it can fit in our GPU global memory.
