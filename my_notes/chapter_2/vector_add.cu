/*
Comment 1 :

 Therefore, you should note the following distinction:
➤ Host: the CPU and its memory (host memory)
➤ Device: the GPU and its memory (device memory)
To help clearly designate the different memory spaces, example code in this book uses variable
names that start with h_ for host memory, and d_ for device memory.

Starting with CUDA 6, NVIDIA introduced a programming model improvement called Unifi ed
Memory, which bridges the divide between host and device memory spaces. This improvement
allows you to access both the CPU and GPU memory using a single pointer, while the system auto-
matically migrates the data between the host and device
*/

/*
Comment 2 : 

The CUDA runtime provides functions to allocate device memory, release device memory, 
and transfer data between the host memory and device memory.

STANDARD C FUNCTIONS    CUDA C FUNCTIONS
malloc                  cudaMalloc
memcpy                  cudaMemcpy
memset                  cudaMemset
free                    cudaFree

To perform GPU memory allocation is : 

cudaError_t cudaMalloc ( void** devPtr, size_t size )

This function allocates a linear range of device memory with the specified size in bytes. The allo-
cated memory is returned through devPtr. You may notice the striking similarity between cuda-
Malloc and the standard C runtime library malloc.

The function used to transfer data between the host and device is: cudaMemcpy, and its function
signature is:

cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count,
cudaMemcpyKind kind )

This function copies the specified bytes from the source memory area, pointed to by src, to the des-
tination memory area, pointed to by dst, with the direction specified by kind, where kind takes one
of the following types:

➤ cudaMemcpyHostToHost
➤ cudaMemcpyHostToDevice
➤ cudaMemcpyDeviceToHost
➤ cudaMemcpyDeviceToDevice

This function exhibits synchronous behavior. 

You can convert an error code to a human-readable error message with the following CUDA run-
time function:

char* cudaGetErrorString(cudaError_t error)

The cudaGetErrorString function is analogous to the Standard C strerror function.
*/

/*
Comment 3 : 

In the GPU memory hierarchy, the two most important types of memory are global
memory and shared memory. Global memory is analogous to CPU system memory,
while shared memory is similar to the CPU cache. However, GPU shared memory
can be directly controlled from a CUDA C kernel

*/
