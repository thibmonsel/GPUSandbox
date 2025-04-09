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

In day 2 will try to minimize the transfer between host and device to improve the overall executing time.