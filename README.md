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

**Summary** : An example of a matrix addition on the host and device. Typically, a matrix is stored linearly in global memory with a row-major approach. 

**Concepts used** :
- Matrix is stored in global memory in row-major approach.  

**TakeAways** : With such a grid and block configuration vector and matrix addition are the same but you have to be careful with indices.