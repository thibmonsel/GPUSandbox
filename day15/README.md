# Day15

Exploring CUDA's Dynamic Parallelism (p.122 from book).

Quote p.122 : 
**Up until now, you had to express your algorithms as individual, massively data parallel kernel launches. Dynamic parallelism enables a more hierarchical approach where concurrency can be expressed in multiple levels in a GPU kernel. Using dynamic parallelism can make your recursive algorithm more transparent and easier to understand. With dynamic parallelism, you can postpone the decision of exactly how many blocks and grids to create on a GPU until runtime, taking advantage of the GPU hardware schedulers and load balancers dynamically and adapting in response to data-driven decisions or workloads. The ability to create work directly from the GPU can also reduce the need to transfer execution control and data between the host and device, as launch configuration decisions can be made at runtime by threads executing on the device**

In dynamic parallelism, kernel executions are classified into two types: parent and child. A parent thread, parent thread block, or parent grid has launched a new grid, the child grid. A child thread, child thread block, or child grid has been launched by a parent. A child grid must complete before the parent thread, parent thread block, or parent grids are considered complete. A parent is not considered complete until all of its child grids have completed.

Grid launches in a device thread are visible across a thread block. This means that a thread may synchronize on the child grids launched by that thread or by other threads in the same thread block. Execution of a thread block is not considered complete until all child grids created by all threads in the block have completed. If all threads in a block exit before all child grids have completed, implicit synchronization on those child grids is triggered. 

When a parent launches a child grid, the child is not guaranteed to begin execution until the parent thread block explicitly synchronizes on the child.

To launch :

```bash
nvcc -rdc=true dynamicParallelism.cu -o dynamicParallelism -lcudadevrt
```

WARNING : 

As dynamic parallelism is supported by the device runtime library, `dynamicParallelism` must be
explicitly linked with `-lcudadevrt` on the command line.

The flag `-rdc=true` forces the generation of relocatable device code, a requirement for dynamic parallelism.

Profiling `dynamicParallelism.cu` : 

```bash
./dynamicParallelism 
./dynamicParallelism Execution Configuration: grid 1 block 8
Recursion=0: Hello World from thread 0 block 0
Recursion=0: Hello World from thread 1 block 0
Recursion=0: Hello World from thread 2 block 0
Recursion=0: Hello World from thread 3 block 0
Recursion=0: Hello World from thread 4 block 0
Recursion=0: Hello World from thread 5 block 0
Recursion=0: Hello World from thread 6 block 0
Recursion=0: Hello World from thread 7 block 0
-------> nested execution depth: 1
Recursion=1: Hello World from thread 0 block 0
Recursion=1: Hello World from thread 1 block 0
Recursion=1: Hello World from thread 2 block 0
Recursion=1: Hello World from thread 3 block 0
-------> nested execution depth: 2
Recursion=2: Hello World from thread 0 block 0
Recursion=2: Hello World from thread 1 block 0
-------> nested execution depth: 3
Recursion=3: Hello World from thread 0 block 0
```