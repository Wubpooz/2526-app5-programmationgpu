#include <cstdio>  
#include <iostream>
#include "cuda.h"  

#define N 512

// A and C are stored by rows, i.e., A(i, j) = A[i * N + j], C(i, j) = C[i * N + j]
// B is stored by columns, i.e., B(i, j) = B[i + j * N]
float *A, *B, *C;

// dA and dC are stored by rows, dB is stored by columns
float *dA, *dB, *dC;


// Create a block for computing each element C(i, j), compute using 1 thread by block
__global__ void multiplyMatrixGPUByBlocks(float *dA, float *dB, float *dC, int n)
{
  int i = blockIdx.x; // i increases by 1 for each block in the x dimension
  int j = blockIdx.y; // j increases by 1 for each block in the y dimension
  dC[i * n + j] = 0.0f;
  for (int k = 0; k < n; k++) {
    dC[i * n + j] += dA[i * n + k] * dB[k + j * n];
  }
}


// Create a block for computing blockDim.x elements of C, compute using blockDim.x threads per block. Each thread computes one element of C
// Assume N is a multiple of blockDim.x
__global__ void multiplyMatrixGPUByBlocksThreads1D(float *dA, float *dB, float *dC, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // i increases by 1 for each thread in a block, then by blockDim.x for each block
  int j = blockIdx.y; // = 1
  dC[i * n + j] = 0.0f;
  for (int k = 0; k < n; k++) {
    dC[i * n + j] += dA[i * n + k] * dB[k + j * n];
  }
}


// Create a block for computing blockDim.x elements of C, compute using blockDim.x threads per block. Each thread computes one element of C
// Make it work when N is not a multiple of blockDim.x
__global__ void multiplyMatrixGPUByBlocksThreads1DNonMultiple(float *dA, float *dB, float *dC, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  if (i < n && j < n) {  // guard to avoid out-of-bounds access because N may not be a multiple of blockDim.x
    dC[i * n + j] = 0.0f;
    for (int k = 0; k < n; k++) {
      dC[i * n + j] += dA[i * n + k] * dB[k + j * n];
    }
  }
}


// Create a block for computing blockDim.x * blockDim.y elements of C, compute using blockDim.x * blockDim.y threads per block.
// Each thread computes one element of C.
// Assume N is a multiple of blockDim.x
__global__ void multiplyMatrixGPUByBlocksThreads2D(float *dA, float *dB, float *dC, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // i increases by 1 for each thread in a block, then by blockDim.x for each block in the x dimension
  int j = blockIdx.y * blockDim.y + threadIdx.y; // j increases by 1 for each thread in a block, then by blockDim.y for each block in the y dimension
  dC[i * n + j] = 0.0f;
  for (int k = 0; k < n; k++) {
    dC[i * n + j] += dA[i * n + k] * dB[k + j * n];
  }
}


// Create a block for computing blockDim.x * blockDim.y elements of C, compute using blockDim.x * blockDim.y threads per block. Each thread computes one element of C
// Make it work when N is not a multiple of blockDim.x nor blockDim.y
__global__ void multiplyMatrixGPUByBlocksThreads2DNonMultiple(float *dA, float *dB, float *dC, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) { // guard to avoid out-of-bounds access because N may not be a multiple of blockDim.x nor blockDim.y
    dC[i * n + j] = 0.0f;
    for (int k = 0; k < n; k++) {
      dC[i * n + j] += dA[i * n + k] * dB[k + j * n];
    }
  }
}


// Reference CPU code for multipying matrices C = AB (A, C stored by rows, B stored by columns)
void multiplyMatrixCPU()
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0.0f;
      for (int k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k + j * N];
      }
    }
  }
}

void verifyResults()
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float c = 0.0f;
      for (int k = 0; k < N; k++) {
        c += A[i * N + k] * B[k + j * N];
      }
      if (std::abs(C[i * N + j] - c) > 1e-6) {
        std::cout << "Multiplication is incorrect for the element C[" << i << "][" << j << "]" << std::endl;
        return;
      }
    }
  }
  std::cout << "Multiplication is correct!" << std::endl;
}

int main(int argc, char **argv)
{
  // Initialization
  A = (float *)malloc(N * N * sizeof(A[0]));
  B = (float *)malloc(N * N * sizeof(B[0]));
  C = (float *)malloc(N * N * sizeof(C[0]));
  for (int j = 0; j < N; j++) { 
    for (int i = 0; i < N; i++) { 
      A[i + j * N] = i + j; // A(i, j) = i + j
      B[i + j * N] = 1.0f; // B(j, i) = 1
    }
  }

  // Allocate dA and dB, then copy the arrays A and B to the GPU
  cudaMalloc(&dA, sizeof(float) * N * N);
  cudaMalloc(&dB, sizeof(float) * N * N);
  cudaMalloc(&dC, sizeof(float) * N * N);

  cudaMemcpy(dA, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

  // Call each GPU kernel appropriately to multiply matrices A and B
  // Measure and print the execution time and performance (GFlops/s) of each kernel, without counting the data transfer time
  {
    dim3 dimGrid(N, N); // one block for each element of C
    dim3 dimBlock; // one thread per block
    multiplyMatrixGPUByBlocks<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }
  {
    dim3 dimGrid(N);
    dim3 dimBlock(N);
    multiplyMatrixGPUByBlocksThreads1D<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }
  { 
    dim3 dimGrid(N);
    dim3 dimBlock(N);
    multiplyMatrixGPUByBlocksThreads1DNonMultiple<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }
  {
    dim3 dimGrid(N, N);
    dim3 dimBlock(N, N);
    multiplyMatrixGPUByBlocksThreads2D<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }
  {
    dim3 dimGrid(N, N);
    dim3 dimBlock(N, N);
    multiplyMatrixGPUByBlocksThreads2DNonMultiple<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
  }

  // Copy the array dC back to the CPU
  cudaMemcpy(C, dC, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

  // Verify the results
  multiplyMatrixCPU();
  verifyResults();

  // Deallocate A, B, C
  free(A); free(B); free(C);

  // Deallocate dA, dB, dC
  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  return 0;
}
