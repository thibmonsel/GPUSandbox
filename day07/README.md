# Day07

Exploring warp divergence and branch efficiency in CUDA.

```bash 
nvprof ./simpleDivergence
Array size: 33554432 elements (134217728 bytes)
Workload iterations per branch: 500
Grid: (131072, 1, 1), Block: (256, 1, 1)
Allocating Device Memory (134217728 bytes)...
==2183087== NVPROF is profiling process 2183087, command: ./simpleDivergence

Running warmup kernel (mathKernel2)...
Warmup <<< 131072,  256 >>> elapsed 216.588 ms

Running mathKernel1 (High Intra-Warp Divergence)...
mathKernel1 <<< 131072,  256 >>> elapsed 316.538 ms (Expect higher time)

Running mathKernel2 (Inter-Warp Divergence Only)...
mathKernel2 <<< 131072,  256 >>> elapsed 161.006 ms (Expect lower time)

Running mathKernel3 (Separate Ifs, High Intra-Warp Divergence)...
mathKernel3 <<< 131072,  256 >>> elapsed 302.839 ms (Expect time similar to Kernel 1)

Cleaning up resources...
Done
==2183087== Profiling application: ./simpleDivergence
==2183087== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   37.74%  377.51ms         2  188.76ms  161.00ms  216.52ms  mathKernel2(float*)
                   31.64%  316.53ms         1  316.53ms  316.53ms  316.53ms  mathKernel1(float*)
                   30.27%  302.83ms         1  302.83ms  302.83ms  302.83ms  mathKernel3(float*)
                    0.35%  3.4685ms         4  867.12us  864.16us  870.05us  [CUDA memset]
      API calls:   86.21%  996.88ms         4  249.22ms  161.00ms  316.53ms  cudaEventSynchronize
                   13.10%  151.44ms         1  151.44ms  151.44ms  151.44ms  cudaMalloc
                    0.35%  4.0685ms       114  35.688us      45ns  2.3548ms  cuDeviceGetAttribute
                    0.23%  2.6120ms         3  870.65us  867.91us  872.42us  cudaDeviceSynchronize
                    0.08%  920.97us         4  230.24us  4.2440us  904.95us  cudaLaunchKernel
                    0.02%  237.93us         1  237.93us  237.93us  237.93us  cudaFree
                    0.00%  54.205us         4  13.551us  4.6500us  29.144us  cudaMemset
                    0.00%  32.741us         8  4.0920us  2.2410us  11.215us  cudaEventRecord
                    0.00%  23.916us         1  23.916us  23.916us  23.916us  cuDeviceGetName
                    0.00%  8.5680us         2  4.2840us     318ns  8.2500us  cudaEventCreate
                    0.00%  6.0770us         1  6.0770us  6.0770us  6.0770us  cuDeviceGetPCIBusId
                    0.00%  5.9860us         4  1.4960us     982ns  2.1050us  cudaEventElapsedTime
                    0.00%  1.9730us         2     986ns     251ns  1.7220us  cudaEventDestroy
                    0.00%  1.7580us         4     439ns      61ns  1.5410us  cudaGetLastError
                    0.00%  1.4340us         3     478ns      65ns  1.2540us  cuDeviceGetCount
                    0.00%     445ns         2     222ns      49ns     396ns  cuDeviceGet
                    0.00%     183ns         1     183ns     183ns     183ns  cuDeviceTotalMem
                    0.00%     101ns         1     101ns     101ns     101ns  cuModuleGetLoadingMode
                    0.00%      76ns         1      76ns      76ns      76ns  cuDeviceGetUuid
```
Warp divergence is clearly visible in the execution time (`mathKernel1` and `mathKernel3` reduces time execution by 50%).

You can force CUDA to not optimize your branch predictions with the following command.

```bash
nvcc -g -G simpleDivergence.cu -o simpleDivergence
```

You can some statistic of the warp divergence with NSight Compute CLI :

```bash
ncu --metrics \
    smsp__inst_executed_op_branch,\
smsp__warps_issue_stalled_branch_resolving,\
smsp__average_warp_latency_issue_stalled_branch_resolving,\
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active,\
smsp__warp_issue_stalled_branch_resolving_per_warp_active \
    ./simpleDivergence
```

which yields : 

```bash
[2189057] simpleDivergence@127.0.0.1
  mathKernel2(float *) (131072, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    ---------------------------------------------------------------------------- ----------- --------------
    Metric Name                                                                  Metric Unit   Metric Value
    ---------------------------------------------------------------------------- ----------- --------------
    smsp__average_warp_latency_issue_stalled_branch_resolving.max_rate             inst/warp              1
    smsp__average_warp_latency_issue_stalled_branch_resolving.pct                          %     760,092.49
    smsp__average_warp_latency_issue_stalled_branch_resolving.ratio                inst/warp       7,600.92
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.max_rate        inst              1
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.pct                %          89.70
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio           inst           0.90
    smsp__inst_executed_op_branch.avg                                                   inst     22,282,240
    smsp__inst_executed_op_branch.max                                                   inst     22,284,064
    smsp__inst_executed_op_branch.min                                                   inst     22,281,560
    smsp__inst_executed_op_branch.sum                                                   inst  1,247,805,440
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.max_rate                                    1
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct                          %           9.00
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.ratio                                    0.09
    smsp__warps_issue_stalled_branch_resolving.avg                                      warp 142,324,061.05
    smsp__warps_issue_stalled_branch_resolving.max                                      warp    142,345,193
    smsp__warps_issue_stalled_branch_resolving.min                                      warp    142,303,539
    smsp__warps_issue_stalled_branch_resolving.sum                                      warp  7,970,147,419
    ---------------------------------------------------------------------------- ----------- --------------

  mathKernel1(float *) (131072, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    ---------------------------------------------------------------------------- ----------- --------------
    Metric Name                                                                  Metric Unit   Metric Value
    ---------------------------------------------------------------------------- ----------- --------------
    smsp__average_warp_latency_issue_stalled_branch_resolving.max_rate             inst/warp              1
    smsp__average_warp_latency_issue_stalled_branch_resolving.pct                          %   1,547,325.51
    smsp__average_warp_latency_issue_stalled_branch_resolving.ratio                inst/warp      15,473.26
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.max_rate        inst              1
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.pct                %          95.10
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio           inst           0.95
    smsp__inst_executed_op_branch.avg                                                   inst  44,527,030.86
    smsp__inst_executed_op_branch.max                                                   inst     44,530,428
    smsp__inst_executed_op_branch.min                                                   inst     44,525,672
    smsp__inst_executed_op_branch.sum                                                   inst  2,493,513,728
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.max_rate                                    1
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct                          %           9.71
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.ratio                                    0.10
    smsp__warps_issue_stalled_branch_resolving.avg                                      warp 289,730,070.20
    smsp__warps_issue_stalled_branch_resolving.max                                      warp    289,773,011
    smsp__warps_issue_stalled_branch_resolving.min                                      warp    289,692,337
    smsp__warps_issue_stalled_branch_resolving.sum                                      warp 16,224,883,931
    ---------------------------------------------------------------------------- ----------- --------------

  mathKernel2(float *) (131072, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    ---------------------------------------------------------------------------- ----------- --------------
    Metric Name                                                                  Metric Unit   Metric Value
    ---------------------------------------------------------------------------- ----------- --------------
    smsp__average_warp_latency_issue_stalled_branch_resolving.max_rate             inst/warp              1
    smsp__average_warp_latency_issue_stalled_branch_resolving.pct                          %     760,092.73
    smsp__average_warp_latency_issue_stalled_branch_resolving.ratio                inst/warp       7,600.93
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.max_rate        inst              1
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.pct                %          89.70
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio           inst           0.90
    smsp__inst_executed_op_branch.avg                                                   inst     22,282,240
    smsp__inst_executed_op_branch.max                                                   inst     22,284,064
    smsp__inst_executed_op_branch.min                                                   inst     22,281,560
    smsp__inst_executed_op_branch.sum                                                   inst  1,247,805,440
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.max_rate                                    1
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct                          %           9.00
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.ratio                                    0.09
    smsp__warps_issue_stalled_branch_resolving.avg                                      warp 142,324,105.23
    smsp__warps_issue_stalled_branch_resolving.max                                      warp    142,349,273
    smsp__warps_issue_stalled_branch_resolving.min                                      warp    142,298,699
    smsp__warps_issue_stalled_branch_resolving.sum                                      warp  7,970,149,893
    ---------------------------------------------------------------------------- ----------- --------------

  mathKernel3(float *) (131072, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: Command line profiler metrics
    ---------------------------------------------------------------------------- ----------- --------------
    Metric Name                                                                  Metric Unit   Metric Value
    ---------------------------------------------------------------------------- ----------- --------------
    smsp__average_warp_latency_issue_stalled_branch_resolving.max_rate             inst/warp              1
    smsp__average_warp_latency_issue_stalled_branch_resolving.pct                          %   1,551,774.19
    smsp__average_warp_latency_issue_stalled_branch_resolving.ratio                inst/warp      15,517.74
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.max_rate        inst              1
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.pct                %          96.82
    smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio           inst           0.97
    smsp__inst_executed_op_branch.avg                                                   inst  42,261,357.71
    smsp__inst_executed_op_branch.max                                                   inst     42,264,582
    smsp__inst_executed_op_branch.min                                                   inst     42,260,068
    smsp__inst_executed_op_branch.sum                                                   inst  2,366,636,032
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.max_rate                                    1
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct                          %           9.75
    smsp__warp_issue_stalled_branch_resolving_per_warp_active.ratio                                    0.10
    smsp__warps_issue_stalled_branch_resolving.avg                                      warp 290,563,067.48
    smsp__warps_issue_stalled_branch_resolving.max                                      warp    290,608,176
    smsp__warps_issue_stalled_branch_resolving.min                                      warp    290,527,432
    smsp__warps_issue_stalled_branch_resolving.sum                                      warp 16,271,531,779
    ---------------------------------------------------------------------------- ----------- --------------
```

<br>
<br />

Key metrics here are :

- `smsp__inst_executed_op_branch.sum`: Total number of branch instructions executed by the SMs (Streaming Multiprocessors) for this kernel launch.
- `smsp__warps_issue_stalled_branch_resolving.sum`: A measure of how many cycles (or similar events) warps spent stalled waiting for branch destinations to be determined. High values indicate significant branching overhead, often related to divergence.
- `smsp__average_warp_latency_issue_stalled_branch_resolving.ratio`: Average number of cycles a warp was stalled waiting for branch resolution per stalled instance.

<br>
<br />

1) `mathKernel1` (High Intra-Warp Divergence):

    - `inst_executed_op_branch.sum`: ~2.49 Billion
    - `warps_issue_stalled_branch_resolving.sum`: ~16.22 Billion
    - `average_warp_latency...ratio`: ~15,473

2) `mathKernel2` (Timed - No Intra-Warp Divergence):

    - `inst_executed_op_branch.sum`: ~1.25 Billion
    - `warps_issue_stalled_branch_resolving.sum`: ~7.97 Billion
    - `average_warp_latency...ratio`: ~7,600

3) `mathKernel3` (High Intra-Warp Divergence):
    - `inst_executed_op_branch.sum`: ~2.37 Billion
    - `warps_issue_stalled_branch_resolving.sum`: ~16.27 Billion
    - `average_warp_latency...ratio`: ~15,517

<br>
<br />


Quotes p.80 :

*Warps are the basic unit of execution in an SM. When you launch a grid of thread blocks, the thread blocks in the grid are distributed among SMs. Once a thread block is scheduled to an SM, threads in the thread block are further partitioned into warps. A warp consists of 32 consecutive threads and all threads in a warp are executed in Single Instruction Multiple Thread (SIMT) fashion; that is, all threads execute the same instruction, and each thread carries out that operation on its own private data*

The number of warps for a thread block can be determined as follows: 

$$\text{WarpsPerBlock} = \text{ceil} \left(\frac{\text{ThreadsPerBlock}}{\text{warpSize}}\right)$$

Thus, the hardware always allocates a discrete number of warps for a thread block. A warp is never split between different thread blocks. If thread block size is not an even multiple of warp size, some threads in the last warp are left inactive. 

Knowning about warps are an important part in designing your kernels. Something called warp divergence can happen when using control flow constructs (`if`, `else`, `while`).

**Because all threads in a warp must execute identical instructions on the same cycle, if one thread executes an instruction, all threads in the warp must execute that instruction. This could become a problem if threads in the same warp take different paths through an application.** 

Consider the following code :
```c
if (cond) {
...
} else {
...
}
```

Suppose 16 threads in the warps executed this code, `cond` is `true`, but for the other 16 `cond` is `false`. Threads in the same warp executing
different instructions is referred to as **warp divergence.** Warp divergence would seem to cause a
paradox, as you already know that all threads in a warp must execute the same instruction on each cycle.

If threads of a warp diverge, the warp serially executes each branch path, disabling threads that do not take that path. Warp divergence can cause significantly degraded performance. In the preceding example, the amount of parallelism in the warp was cut by half: only 16 threads were actively executing at a time while the other 16 were disabled. With more conditional branches, the loss of parallelism would be even greater. (Take note that branch divergence occurs only within a warp. Different conditional values in differ-
ent warps do not cause warp divergence).


- Intra-Warp Divergence (Bad): Threads within the same warp disagree on a branch condition. Hardware serializes the different paths, masking inactive threads. Reduces parallelism within the warp. Performance degrades.

- Inter-Warp Differences (Fine): Different warps make different branch decisions based on their data. Each warp executes its chosen path efficiently (assuming no internal divergence within that warp). This is normal, expected parallel execution across independent warps. No serialization penalty between warps occurs due to these differing paths.