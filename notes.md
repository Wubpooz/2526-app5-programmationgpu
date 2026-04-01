# Notes Cours

**Table of Contents**  
- [Notes Cours](#notes-cours)
  - [Architecture](#architecture)
  - [Cuda](#cuda)
    - [Qualifiers](#qualifiers)
      - [Functions](#functions)
      - [Variables](#variables)
    - [Special Functions](#special-functions)
      - [Static Memory](#static-memory)
      - [Dynamic Memory](#dynamic-memory)
      - [Synchronization](#synchronization)



## Architecture
GPU architecture is SIMT (Single Instruction Multiple Threads) : a single instruction is executed by multiple threads in parallel. A GPU is made of multiple streaming multiprocessors (SMs), each made of 32 cores. In each SM we can only put up to 1024 threads. A thread is the execution of a kernel (GPU functions) on a core. It is identified by a `threadIdx.x`.     
A block is the execution of a kernel (GPU functions) on a SM. It is identified by a `blockIdx.x`.  


&nbsp;  
## Cuda
Compiler: `nvcc`.  
Library: `cuda.h`.   
It can be compiled and run in [Godbolt](https://godbolt.org/).  

### Qualifiers
#### Functions 
- `__host__` (default) : function called and executed on the CPU (host).
  Returns any type.
  Executed by a **single thread**.
- `__global__` : function called from the CPU and executed on the GPU (device).
  **Must return void**.
  Executed by a **grid of threads**.
- `__device__` : function called and executed on the GPU (device).
  Returns any type.
  Executed by a **single thread**.

#### Variables  
- `__device__` : GPU's global memory.
  Accessible by all threads.
  Lives as long as the program is running.
- `__shared__` : GPU's shared memory.
  Accessible by all threads **of a block**.
  Lives as long as the **block** is running.
- `__constant__` : GPU's constant memory.
  Accessible by all threads.
  Read-only and can only be initialized by the CPU.

All are initialized to 0 by default.


&nbsp;  
### Special Functions
#### Static Memory
- `cudaMemcpyToSymbol(void *symbol, const void *src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice)` : copies `count` bytes from the memory area pointed to by `src` to the memory area pointed to by `symbol` on the device. The `offset` parameter specifies the byte offset from the start of the symbol.
- `cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost)` : copies `count` bytes from the memory area pointed to by `symbol` on the device to the memory area pointed to by `dst` on the host. The `offset` parameter specifies the byte offset from the start of the symbol.

#### Dynamic Memory
- `cudaMalloc(void **devPtr, size_t size)` : allocates `size` bytes of memory on the GPU and returns a pointer to the allocated memory in `devPtr`.
- `cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)` : copies `count` bytes from the memory area pointed to by `src` to the memory area pointed to by `dst`. The `kind` parameter specifies the direction of the copy (host to device, device to host, or device to device).
- `cudaFree(void *devPtr)` : frees the memory space pointed to by `devPtr`, which must have been returned by a previous call to `cudaMalloc()`.


#### Synchronization
- `__syncthreads()`: synchronizes all threads in a block. It ensures that all threads have reached this point before any thread continues. It is used to avoid race conditions