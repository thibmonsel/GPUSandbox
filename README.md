# GPUSandbox

Repository to explore CUDA C programming language and build GPU kernels for 100 days ! 

This repository contain my personal notes and code while going through the open-access [book](https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf) (code located [here](https://github.com/deeperlearning/professional-cuda-c-programming/tree/master)) in `my_notes` : 

# Learning GPU kernels for 100 days

## Day 1

### File `vectorAdd.cu`

**Summary** : An example of a vector addition on the host and device. Using a simple kernel to add two
vectors together. The kernel is launched with a grid of blocks, where each block contains a
number of threads. Each thread is responsible for adding one element of the two vectors
together.

**Concepts used** :
- Allocated memory for both the host and device (i.e. GPU).
- Learned how threads, blocks and grid work in CUDA (for more information, checkout [here](https://harmanani.github.io/classes/csc447/Notes/Lecture15.pdf)).
- Launched a kernel with `__global__` keyword.
- Timed our execution with `cudaEventCreate`,`cudaEventRecord` and `cudaEventSynchronize`.
- Managed the deallocation of the vectors.

**TakeAways** : We saw that in such a case memory transfer from Host to Device (HtD) and Device to Host (DtH) takes a majority of the execution time (90%). 

## Day 2

### File `vectorAddWithStreams.cu`

**Summary** : An example of a vector addition on the host and device. Using a simple kernel to add two
vectors together. Day 2 is basically the same as day 1 but with CUDA streams to decrease the data transfer HtD and DtH.

**Concepts used** :
- Used CUDA streams to load data asynchronously.
- Split the data according to the number of streams used.
- Delved into memory transfer between the host and device.

**TakeAways** : Thanks to CUDA streams that leverages asynchronous data transfer the global execution was greatly reduced.


## Day 3

### File `vectorAddOptimized.cu`

**Summary** : An example of a vector addition on the host and device. Using a simple kernel to add two
vectors together that leverage CUDA streams and the CUDA type `float4`. Day 3 is basically the same as day 3 but with the type `float4`.

**Concepts used** :
- Learn about `float4` CUDA type
- Saw a grid-stride loop over a `float4` elements (each thread now has more operations to do).

**TakeAways** : CUDA type `float4` use one memory instruction for loading 4 floats. Grid-stride loops allow to reduce overhead, allows for instruction level parallelism (ILP), i.e. more independent instructions within a thread.

## Day 4

### File `matrixAdd.cu`

**Summary** : An example of a matrix addition on the host and device. Typically, a matrix is stored linearly in global memory with a row-major approach. Using the row-major approach to assimilate the matrix multiplication as a 1D vector addition. 

**Concepts used** :
- Matrix is stored in global memory in row-major approach.  

**TakeAways** : With such a grid and block configuration vector and matrix addition are the same but you have to be careful with indices.

## Day 5

### File `matrixAdd2.cu`

**Summary** : An example of a matrix addition on the host and device. Using a 1D block and 1D grid configuration, where each block is responsible for a chunk of matrix columns.

**TakeAways** : Used another configuration grid, block configuration for the same operation as in day 04. 


## Day 6

### File `matrixAdd3.cu`

**Summary** : An example of a matrix addition on the host and device. Using a 2D block and 2D grid configuration, where each block is responsible for a matrix chunk.

**TakeAways** : Used another configuration grid, block configuration for the same operation as in day 05.

## Day 7

### File `simpleDivergence.cu`

**Summary** : An example of warp divergence for CUDA and how to deal with it.

**Concepts used** :
- Issues with control-flow instructions in CUDA.
- Intra-Warp divergence degrades performance.
- Inter-War divergence (called Inter-Warp Variation) doesn't degrade performance because different warps make different branch decisions based on their data.

**TakeAways** : Intra-Warp Divergence will make your code drop in performance. 

# Day 8

### File `latencyHiding.cu`

**Summary** : An example of the two types of arithmetic and memory latency hiding

**Concepts used** :
- Discussed the "Occupancy" metric that shows how utilized is the hardware.
- Inspected arithmetic and memory latency hiding.
- Computed the number of warps needed to get latency hiding for a given block and grid configuration.

**TakeAways** : To hide arithmetic and memory latency a balance between grid and block size needs to be done. 
- Small thread blocks: Too few threads per block leads to hardware limits on the number of warps per SM to be reached before all resources are fully utilized.
- Large thread blocks: Too many threads per block leads to fewer per-SM hardware resources available to each thread.

# Day 9

### File `synchronize.cu`

**Summary** : An example of the thread and global synchronization

**Concepts used** :
- Used ` __syncthreads();` and `cudaDeviceSynchronize()`.
- Very briefly saw the concept of "sticky-errors" in CUDA (this means that the CUDA context is corrupted). Great link [here](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/) !

**TakeAways** : CUDA kernel launches are asynchronous. `cudaDeviceSynchronize()` is crucial for host-device coordination and for reliably catching kernel runtime errors, while `__syncthreads()` manages synchronization within a thread block. Beware of sticky errors impacting subsequent operations.

# Day 10

### File `parallelReduce.cu`

**Summary** : An example of the parallel reduction operator in CUDA.

**Concepts used** :
- Discussed and implemented the parallel reduced with a neighbored pair method.
- Warp divergence is present in our problem 

**TakeAways** : The parallel reduction operator is a fundamental technique in CUDA for efficiently combining elements of an array (e.g., summing, finding min/max) in parallel.

# Day 11

### File `parallelReduce2.cu`

**Summary** : Second example of the parallel reduction operator in CUDA.

**Concepts used** :
- Reduced warp divergence is present in our problem 

**TakeAways** : We changed the parallel reduction implementation by leverage different thread indexing to reduce warp divergence.


# Day 12

### File `parallelReduce3.cu`

**Summary** : Third example of the parallel reduction operator in CUDA.

**Concepts used** :
- Implemented Interleaved based parallel reduction method 
- Same amount of warp divergence as in `day11`.

**TakeAways** : This method is more optimal due to the better memory management with coalescing memory banks. Compared to day12 we saw a 2x improvement.