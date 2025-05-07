# Day 09

No profiling hear to do but more of a broad understanding of synchronization (global and thread level). 

```bash
./synchronize 
Using CUDA Device 0: NVIDIA T600 Laptop GPU
Compute Capability: 7.5

--- Test 1: cudaDeviceSynchronize Error Reporting ---
Launching faulty_kernel asynchronously...
Faulty Kernel: Thread 0 attempting to write to an invalid address out_data[N+10]. This should cause an error.
Calling cudaDeviceSynchronize() to wait for completion and catch runtime errors...
NOTE: cudaDeviceSynchronize() did NOT report an error. Checking if a previous cudaGetLastError() did 
 and no prior launch config error was found. This is unexpected for this faulty_kernel setup.
No further host-side sticky error found by final cudaGetLastError() in Test 1.
--- Test 1 Finished (Note: The CUDA device/context might be unstable due to the intentional error) ---

--- Test 2: __syncthreads for Block-Level Synchronization & Shared Memory ---
Launching syncthreads_demo_kernel (block size: 256, grid size: 64)...
Calling cudaDeviceSynchronize() to ensure kernel completion...
SUCCESS: __syncthreads demo kernel results verified.
Example block sum (block 0): 1140
--- Test 2 Finished ---

--- Test 3: Global Synchronization by Kernel Termination ---
Launching first_phase_kernel...
Calling cudaDeviceSynchronize() to ensure first_phase_kernel completes globally.
Launching second_phase_kernel which depends on first_phase_kernel's output...
Calling cudaDeviceSynchronize() to ensure second_phase_kernel completes.
SUCCESS: Global synchronization test passed. Example output[0]=200 (expected 200).
--- Test 3 Finished ---

All tests completed.
```


Quote p.97: 

*Synchronization can be performed at two levels:*
*➤ System-level: Wait for all work on both the host and the device to complete.*
*➤ Block-level: Wait for all threads in a thread block to reach the same point in execution on the device.*

*Since many CUDA API calls and all kernel launches are asynchronous with respect to the host, cudaDeviceSynchronize can be used to block the host application until all CUDA operations (copies, kernels, and so on) have completed*

*`cudaError_t cudaDeviceSynchronize(void);`*

*This function may return errors from previous asynchronous CUDA operations.*

*Because warps in a thread block are executed in an undefi ned order, CUDA provides the ability to synchronize their execution with a block-local barrier. You can mark synchronization points in the kernel using:*

*`__device__ void __syncthreads(void);`*

*Threads within a thread block can share data through shared memory and registers. When sharing data between threads you need to be careful to avoid race conditions. Race conditions, or hazards, are unordered accesses by multiple threads to the same memory location. For example, a read-after- write hazard occurs when an unordered read of a location occurs following a write. Because there is no ordering between the read and the write, it is undefi ned if the read should have loaded the value of that location before the write or after the write.*