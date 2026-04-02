# Notes Cours

**Table of Contents**  
- [Notes Cours](#notes-cours)
  - [Architecture](#architecture)
  - [Cuda](#cuda)
    - [Multi-dimensional grids/blocks and coalescence](#multi-dimensional-gridsblocks-and-coalescence)
      - [Coalescence](#coalescence)
    - [Qualifiers](#qualifiers)
      - [Functions](#functions)
      - [Variables](#variables)
    - [Special Functions](#special-functions)
      - [Static Memory](#static-memory)
      - [Dynamic Memory](#dynamic-memory)
      - [Synchronization](#synchronization)

&nbsp;  

---

&nbsp;  
## Architecture
GPU architecture is **SIMT** (Single Instruction Multiple Threads): a single instruction is executed by multiple threads in parallel. A GPU is made of multiple streaming multiprocessors (**SMs**), each made of 32 cores. In each SM we can only put up to **1024 threads**. A thread is the execution of a kernel (GPU functions) on a core. It is identified by a `threadIdx.x`.     
A block is the execution of a kernel (GPU functions) on a SM. It is identified by a `blockIdx.x`.  
GPU supports **speculative execution**, which means that it can execute instructions out of order to hide memory latency. It also supports warp divergence, which means that if threads in a warp (group of 32 threads) take different execution paths, they will be executed sequentially, which can lead to performance degradation (a if on `threadIdx.x % 2 == 0` causes divergence and **multiples the execution time** but on `threadIdx.x < SIZE /2` does not).

---

## Cuda
Compiler: `nvcc`.  
Library: `cuda.h`.   
It can be compiled and run in [Godbolt](https://godbolt.org/).  


&nbsp;  
### Multi-dimensional grids/blocks and coalescence
`dim3` is a structure that represents the dimensions of a grid or a block. It has three members: `x`, `y`, and `z`. By default, they are all set to 1.  
It is used to s**pecify the number of blocks and threads in each dimension** when launching a kernel. For example, `dim3 blockSize(256);` or `dim3 grid(256, 1, 1);` means that each block will have 256 threads in the x-dimension, and 1 thread in the y and z dimensions. The total number of threads per block is `blockSize.x * blockSize.y * blockSize.z`.  

&nbsp;  
When launching a kernel, we can **specify the number of blocks and the number of threads per block** using the syntax `kernel<<<numBlocks, numThreadsPerBlock>>>(args)`. The total number of threads launched will be `numBlocks * numThreadsPerBlock`.    
- 2D array mapping examples:  
  - row partition: `i = blockIdx.x`, `j = blockIdx.y * blockDim.x + threadIdx.x`
  - col partition: `i = blockIdx.y * blockDim.x + threadIdx.x`, `j = blockIdx.x`
  - bounds check: `if (i < n && j < n)`
- 2D block + 2D thread: `i = blockIdx.y * blockDim.y + threadIdx.y`, `j = blockIdx.x * blockDim.x + threadIdx.x`
  - each block works on `blockDim.x * blockDim.y` submatrix (e.g., 32x32 for `blockSize=1024`)


&nbsp;  
**Strided Memory Access:** Striding means each thread iterates over elements **jumping** by `stride`, e.g. `idx = base + k*stride`.
- stride=1: each thread in a warp accesses adjacent elements => **best coalescence**.
- stride=2: each thread jumps 2 floats per step, a warp may need 2 DRAM segments for 32 floats, reducing efficiency.
- stride=32: one thread picks every 32nd element, in a warp **31/32 of data is not used** (bad coalescence).
Use small stride in the fastest-varying dimension of row-major data to keep accesses contiguous and coalesced. For wide ranges, you may still use striding with loop steps, but try to make each warp touch neighboring addresses at each iteration.   



&nbsp;  
#### Coalescence
- **GPU memory is row-major**. Accessing consecutive `j` in the inner thread dimension gives coalesced loads/stores.  
- Row-major stripe (threadIdx.x varies fastest across columns) is generally better than column-major access for a standard 2D array in global memory.
- Choose block/grid size to **keep threads in a warp on adjacent addresses** and avoid out-of-bounds in last block.
- Using 2D `dim3` avoids complex division/modulus to recover `i,j` indices from flat block IDs.

&nbsp;  

To check coalescence issues, **observe where warp 0 accesses the memory** to quickly identify problems or confirm good behavior. The values of `i` and `j` for warp 0 (threads 0-31) should be contiguous in memory for good coalescence.  
Example:  
- Coalescent: 
  ```cpp
  int i = blockIdx.y * blockDim.y + threadIdx.y; // row index
  int j = blockIdx.x * blockDim.x + threadIdx.x; // column index
  ```
  - For the warp 0, `i = 0 * blockDim.y + 0 = 0` and `j = 0 * blockDim.x + 0..31 = 0..31`, so threads 0-31 access `A[0][0..31]` which is **contiguous** in row-major order, ensuring coalescence.  
  - Then for warp 1 (threads 32-63), `i = 0 * blockDim.y + 1 = 1` and `j = 0 * blockDim.x + 0..31 = 0..31`, so threads 32-63 access `A[1][0..31]`, which is also contiguous, maintaining coalescence across warps.
- Non-coalescent: 
  ```cpp
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row index
  int j = blockIdx.y * blockDim.y + threadIdx.y; // column index
  ```
  - For warp 0, `i = 0 * blockDim.x + 0..31 = 0..31` and `j = 0 * blockDim.y + 0 = 0`, so threads 0-31 access `A[0..31][0]`, which is **not contiguous** in row-major order, leading to poor coalescence.  
  Then it **strides with a stride of 32** in the row dimension, causing each thread in the warp to access memory locations that are 32 rows apart, which is inefficient for coalescence.  




&nbsp;  
Coalescence Examples:  
```cpp
// Coalesced access example (row-major stripe)
__global__ void coal_kernel(float *A, int n) {
    int i = blockIdx.x; // row index
    int j = blockIdx.y * blockDim.x + threadIdx.x; // column index

    // or in 2D
    i = blockIdx.x * blockDim.y + threadIdx.y; // row index
    j = blockIdx.y * blockDim.x + threadIdx.x; // column index

    if (i < n && j < n) {
        A[i][j] = ...; // access A[i][j] in row-major order
    }
}

// Non coalesced access example (column-major stripe)
__global__ void non_coal_kernel(float *A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row index
    int j = blockIdx.y; // column index
    
    // or in 2D
    i = blockIdx.x * blockDim.x + threadIdx.x; // row index
    j = blockIdx.y * blockDim.y + threadIdx.y; // column index


    if (i < n && j < n) {
        A[i][j] = ...; // access A[i][j] in column-major order
    }
}

int blockSize = 1024; // threads per block
dim3 grid;
grid.x = N;
grid.y = N / blockSize; // number of blocks needed to cover N columns
grid.z = 1;
coal_kernel<<<grid, blockSize>>>(A, N);
non_coal_kernel<<<grid, blockSize>>>(A, N);
```

&nbsp;  

---

&nbsp;  
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