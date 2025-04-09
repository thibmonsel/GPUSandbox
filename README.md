# GPUSandbox

Repository to explore CUDA C programming language and build GPU kernels for 100 days ! 

This repository contain my personal notes and code while going through the open-access [book](https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf) in `my_notes` : 

# Learning GPU kernels for 100 days

## Day 1

### File `vectorAdd.cu`

**Summary** : An example of a vector addition on the host and device. Using a simple kernel to add two
vectors together. The kernel is launched with a grid of blocks, where each block contains a
number of threads. Each thread is responsible for adding one element of the two vectors
together.

**Concepts used** :
- Allocated memory for both the host and device (i.e. GPU).
- Learned how threads, blocks and grid work in CUDA.
- Launched a kernel with `__global__`.
- Timed our execution with `cudaEventCreate`,`cudaEventRecord` and `cudaEventSynchronize`.
- Managed the deallocation of the vectors

**TakeAways** : We saw that in such a case memory transfer from Host to Device (HtD) and Device to Host (DtH) takes a majority of the execution time (90%). 

