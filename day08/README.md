# Day 08

Exploring Latency Hiding in CUDA. In order to hide latency, one must have a certain number of required warps given by Little's Law (c.f. quote below).

$$\text{Number of required Warps} =\text{Latency} \times \text{Throughput}$$

Occupancy is an important metric to see if you are utilizing most of your GPU. Instructions are executed sequentially within each CUDA core. When one warp stalls, the SM switches to executing other eligible warps. Ideally, you want to have enough warps to keep the cores of the device occupied. Occupancy is the ratio of active warps to maximum number of warps, per SM.
 
$$ \text{occupancy} = \frac{\text{active warps}}{\text{maximum warps}} $$

where "maximum warps" is given by the hardware.

```bash 
GUIDELINES FOR GRID AND BLOCK SIZE

Using these guidelines will help your application scale on current and future
devices:
➤ Keep the number of threads per block a multiple of warp size (32).
➤ Avoid small block sizes: Start with at least 128 or 256 threads per block.
➤ Adjust block size up or down according to kernel resource requirements.
➤ Keep the number of blocks much greater than the number of SMs to expose
sufficient parallelism to your device.
➤ Conduct experiments to discover the best execution configuration and
resource usage.
```
Profiling `latencyHiding.cu`

```bash
nvprof ./latencyHiding
==2392628== NVPROF is profiling process 2392628, command: ./latencyHiding
Device: NVIDIA T600 Laptop GPU
Compute Capability: 7.5
Max Threads per Block: 1024
Max Threads per SM: 1024
Warps per SM = Max Threads per SM / 32 = 32

--- Testing Arithmetic Latency Hiding ---

Problem Size N = 32768 (Threads = 32768, Blocks = 128)
Elapsed Time: 0.032 ms
Performance:  1029.146 GFLOPS

Problem Size N = 262144 (Threads = 262144, Blocks = 1024)
Elapsed Time: 0.169 ms
Performance:  1547.412 GFLOPS

Problem Size N = 1048576 (Threads = 1048576, Blocks = 4096)
Elapsed Time: 0.652 ms
Performance:  1608.009 GFLOPS

Problem Size N = 8388608 (Threads = 8388608, Blocks = 32768)
Elapsed Time: 5.168 ms
Performance:  1623.052 GFLOPS

Problem Size N = 16777216 (Threads = 16777216, Blocks = 65536)
Elapsed Time: 10.316 ms
Performance:  1626.350 GFLOPS
-------------------------------------------

--- Testing Memory Latency Hiding ---

Problem Size N = 4194304 (Threads = 4194304, Blocks = 16384, 16.00 MB)
Elapsed Time: 0.345 ms
Effective Bandwidth: 90.481 GiB/s

Problem Size N = 16777216 (Threads = 16777216, Blocks = 65536, 64.00 MB)
Elapsed Time: 1.357 ms
Effective Bandwidth: 92.111 GiB/s

Problem Size N = 67108864 (Threads = 67108864, Blocks = 262144, 256.00 MB)
Elapsed Time: 5.253 ms
Effective Bandwidth: 95.180 GiB/s

Problem Size N = 134217728 (Threads = 134217728, Blocks = 524288, 512.00 MB)
Elapsed Time: 10.199 ms
Effective Bandwidth: 98.050 GiB/s

Problem Size N = 268435456 (Threads = 268435456, Blocks = 1048576, 1024.00 MB)
Elapsed Time: 15.315 ms
Effective Bandwidth: 130.588 GiB/s
-----------------------------------------

Done.
==2392628== Profiling application: ./latencyHiding
==2392628== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   79.47%  425.77ms        10  42.577ms  12.768us  240.01ms  [CUDA memcpy HtoD]
                   12.10%  64.847ms        10  6.4847ms  337.28us  15.308ms  memory_latency_kernel(float const *, float*, unsigned long)
                    6.08%  32.566ms        10  3.2566ms  24.289us  10.297ms  arithmetic_latency_kernel(float*, int, unsigned long)
                    2.35%  12.593ms         5  2.5187ms  109.60us  6.8741ms  [CUDA memset]
      API calls:   55.02%  426.86ms        10  42.686ms  27.639us  240.18ms  cudaMemcpy
                   17.75%  137.69ms         4  34.422ms     378ns  137.69ms  cudaEventCreate
                    7.97%  61.854ms        10  6.1854ms  43.705us  22.229ms  cudaDeviceSynchronize
                    7.94%  61.629ms         1  61.629ms  61.629ms  61.629ms  cudaDeviceReset
                    6.28%  48.713ms        10  4.8713ms  26.034us  15.299ms  cudaEventSynchronize
                    3.71%  28.749ms        15  1.9166ms  44.237us  26.925ms  cudaMalloc
                    0.50%  3.9122ms       114  34.317us     137ns  2.2284ms  cuDeviceGetAttribute
                    0.41%  3.1976ms         1  3.1976ms  3.1976ms  3.1976ms  cudaGetDeviceProperties
                    0.33%  2.5955ms        15  173.04us  41.873us  391.88us  cudaFree
                    0.04%  341.37us        20  17.068us  2.4490us  101.16us  cudaLaunchKernel
                    0.02%  143.08us         5  28.615us  24.800us  33.552us  cudaMemset
                    0.01%  112.88us        20  5.6430us  2.0340us  19.712us  cudaEventRecord
                    0.00%  35.499us         1  35.499us  35.499us  35.499us  cuDeviceGetName
                    0.00%  13.655us        10  1.3650us     733ns  3.0890us  cudaEventElapsedTime
                    0.00%  10.780us         4  2.6950us     341ns  6.9750us  cudaEventDestroy
                    0.00%  4.1980us         1  4.1980us  4.1980us  4.1980us  cuDeviceGetPCIBusId
                    0.00%  3.1030us         1  3.1030us  3.1030us  3.1030us  cudaGetDevice
                    0.00%  2.7370us         3     912ns     213ns  2.2030us  cuDeviceGetCount
                    0.00%  2.6100us        20     130ns      42ns     306ns  cudaGetLastError
                    0.00%     798ns         2     399ns     160ns     638ns  cuDeviceGet
                    0.00%     677ns         1     677ns     677ns     677ns  cuModuleGetLoadingMode
                    0.00%     571ns         1     571ns     571ns     571ns  cuDeviceTotalMem
                    0.00%     264ns         1     264ns     264ns     264ns  cuDeviceGetUuid
```

- Arithmetic: Watch the GFLOPS. It should increase as N grows and potentially saturate when enough warps are active to hide the ~10-20 cycle arithmetic latency. The number of warps needed will be significant but less than for memory.

- Memory: Watch the GiB/s. This should increase dramatically as N grows. Hiding the 400-800 cycle memory latency requires a lot of concurrent requests, hence many active warps (potentially exceeding the theoretical maximum occupancy if the kernel is simple, but limited by device resources). You'll likely see the bandwidth approach a significant fraction of your GPU's theoretical peak memory bandwidth for the largest problem sizes.

<br>
<br />

Given the more detailed profiling of Occupancy
```bash
sudo ncu --section Occupancy ./latencyHiding
```

where we get : 
```bash
[2524027] latencyHiding@127.0.0.1
  arithmetic_latency_kernel(float *, int, unsigned long) (128, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        83.40
    Achieved Active Warps Per SM           warp        26.69
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 16.6%                                                                                     
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (83.4%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         


  arithmetic_latency_kernel(float *, int, unsigned long) (1024, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        95.44
    Achieved Active Warps Per SM           warp        30.54
    ------------------------------- ----------- ------------


  arithmetic_latency_kernel(float *, int, unsigned long) (4096, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        96.80
    Achieved Active Warps Per SM           warp        30.97
    ------------------------------- ----------- ------------


  arithmetic_latency_kernel(float *, int, unsigned long) (32768, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        97.29
    Achieved Active Warps Per SM           warp        31.13
    ------------------------------- ----------- ------------


  arithmetic_latency_kernel(float *, int, unsigned long) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            4
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        97.34
    Achieved Active Warps Per SM           warp        31.15
    ------------------------------- ----------- ------------
```

The occupancy and the number of active warps are low with small blocks (83%) and achieve excellent figures when we run the kernel for `<<< (65536, 1, 1)x(256, 1, 1)>>>` 97% of occupancy is achieved (or 31.15/32 active warps per SM).

Quote p.90-93 :

*An SM relies on thread-level parallelism to maximize utilization of its functional units. Utilization is therefore directly linked to the number of resident warps. The number of clock cycles between an instruction being issued and being completed is defined as instruction latency. Full compute resource utilization is achieved when all warp schedulers have an eligible warp at every clock cycle. This ensures that the latency of each instruction can be hidden by issuing other instructions in other resident warps.*

*Compared with C programming on the CPU, latency hiding is particularly important in CUDA programming. CPU cores are designed to minimize latency for one or two threads at a time, whereas GPUs are designed to handle a large number of concurrent and lightweight threads in order to maximize throughput. GPU instruction latency is hidden by computation from other warps.*

*When considering instruction latency, instructions can be classified into two basic types:*

*➤ Arithmetic instructions* \
*➤ Memory instruction*

*Arithmetic instruction latency is the time between an arithmetic operation starting and its output being produced. Memory instruction latency is the time between a load or store operation being issued and the data arriving at its destination. The corresponding latencies for each case are approximately.*

*➤ 10-20 cycles for arithmetic operations* \
*➤ 400-800 cycles for global memory accesses*


*You may wonder how to estimate the number of active warps required to hide latency. Little’s Law can provide a reasonable approximation*

$$\text{Number of required Warps} =\text{Latency} \times \text{Throughput}$$

## Required parallelism for arithmetic operations 

*For arithmetic operations, the required parallelism can be expressed as the number of operations required to hide arithmetic latency. Table 3-3 lists the number of required operations for Fermi and Kepler devices. The arithmetic operation used as an example here is a 32-bit floating-point multiply-add (a + b × c), expressed as the number of operations per clock cycle per SM. The throughput varies for different arithmetic instructions*

*Throughput is specified in number of operations per cycle per SM, and one warp executing one instruction corresponds to 32 operations. Therefore, the required number of warps per SM to maintain full compute resource utilization can be calculated for Fermi GPUs as 640 ÷ 32 = 20 warps. Hence, the required parallelism for arithmetic operations can be expressed as either the number of operations or the number of warps.*

## Required parallelism for memory operations 

*For memory operations, the required parallelism is expressed as the number of bytes per cycle required to hide memory latency.*

*Because memory throughput is usually expressed as gigabytes per second, you need to first convert the throughput into gigabytes per cycle using the corresponding memory frequency. An example Fermi memory frequency (measured on a Tesla C2070) is 1.566 GHz. An example Kepler memory frequency (measured on a Tesla K20) is 1.6 GHz. Because 1 Hz is defined as one cycle per second, you then can convert the bandwidth from gigabytes per second to gigabytes per cycle as follows*

$$ \text{144 GB/Sec ÷ 1.566 GHz ≅ 92 Bytes/Cycle}$$

*Multiplying bytes per cycle by memory latency, you derive the required parallelism for Fermi memory operations at nearly 74 KB of memory I/O in-fl ight to achieve full utilization. This value is for the entire device, not per SM, because memory bandwidth is given for the entire device*

*Connecting these values to warp or thread counts depends on the application. Suppose each thread moves one float of data (4 bytes) from global memory to the SM for computation, you would require 18,500 threads or 579 warps to hide all memory latency on Fermi GPUs:*

$$\text{74 KB ÷ 4 bytes/thread ≅ 18,500 threads}$$
$$ \text{18,500 threads ÷ 32 threads/warp ≅ 579 warps} $$

*The Fermi architecture has 16 SMs. Therefore, you require 579 warps ÷ 16 SMs = 36 warps per SM to hide all memory latency.*