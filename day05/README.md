# Day 05

CUDA matrix addition using a 1D block and 1D grid configuration, where each block is responsible for a chunk of matrix columns.

```bash
Block 0 -> Col 0                       ...     Col threadIdx.x
Block 1 -> Col threadIdx.x + 1         ...     2 * threadIdx.x
Block 2 -> Col 2 * threadIdx.x + 1     ...     3 * threadIdx.x
```

```bash
nvprof ./matrixAdd2
==355589== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.95%  367.86ms         2  183.93ms  163.31ms  204.55ms  [CUDA memcpy HtoD]
                   30.17%  173.53ms         1  173.53ms  173.53ms  173.53ms  [CUDA memcpy DtoH]
                    5.88%  33.829ms         1  33.829ms  33.829ms  33.829ms  matrixAddonDevice(float*, float*, float*, int, int)
      API calls:   79.83%  575.60ms         3  191.87ms  163.41ms  207.49ms  cudaMemcpy
                   19.42%  140.05ms         3  46.682ms  73.390us  139.90ms  cudaMalloc
                    0.58%  4.1915ms       114  36.767us     178ns  2.3590ms  cuDeviceGetAttribute
                    0.14%  973.69us         3  324.56us  284.86us  401.27us  cudaFree
                    0.02%  163.55us         1  163.55us  163.55us  163.55us  cudaLaunchKernel
                    0.01%  39.784us         2  19.892us  9.9500us  29.834us  cudaEventRecord
                    0.00%  31.412us         1  31.412us  31.412us  31.412us  cuDeviceGetName
                    0.00%  9.6770us         2  4.8380us  1.3320us  8.3450us  cudaEventCreate
                    0.00%  4.9120us         1  4.9120us  4.9120us  4.9120us  cuDeviceGetPCIBusId
                    0.00%  3.3670us         3  1.1220us     221ns  2.7590us  cuDeviceGetCount
                    0.00%  2.1960us         1  2.1960us  2.1960us  2.1960us  cudaEventSynchronize
                    0.00%  1.6620us         1  1.6620us  1.6620us  1.6620us  cudaEventElapsedTime
                    0.00%  1.0940us         2     547ns     197ns     897ns  cudaEventDestroy
                    0.00%     996ns         2     498ns     166ns     830ns  cuDeviceGet
                    0.00%     643ns         1     643ns     643ns     643ns  cuDeviceTotalMem
                    0.00%     510ns         1     510ns     510ns     510ns  cuModuleGetLoadingMode
                    0.00%     329ns         1     329ns     329ns     329ns  cuDeviceGetUuid
                    0.00%     322ns         1     322ns     322ns     322ns  cudaGetLastError
```

Compared to day04's implementation, `matrixAddonDevice` is slower. 
Our goal today is to jungle with different grid and block configuration and not speed.