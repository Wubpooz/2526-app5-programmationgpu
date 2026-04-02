#include <cstdio>  
#include <iostream>
#include "cuda.h"  

#define N 1024
#define BSXY 32

// A and C are stored by rows, i.e., A(i, j) = A[i * N + j], C(i, j) = C[i * N + j]
// B is stored by columns, i.e., B(i, j) = B[i + j * N]
float *A, *B, *C;

// dA and dC are stored by rows, dC is stored by columns
float *dA, *dB, *dC;


// Use BSXY == blockDim.x == blockDim.y (square blocks) in this exercise.
// Create one block for computing BSXY * BSXY elements of C, compute using BSXY * BSXY threads per block.
// Each thread computes a single element of C.
// Make it work when N is not divisible by BSXY.
// To perform the multiplication, Operate on matrix tiles of size BSXY * BSXY of A and B using shared memory.
// Accumulate on BSXY * BSXY registers for a tile of C. That is, in each step,
// read a BSXY * BSXY tile of A and B on shared memory, multiply them and
// accumulate on C on registers, then continue with the rest of the tiles
__global__ void multiplyMatrixGPUByBlocksThreads2DNonMultipleSharedMemory(float *dA, float *dB, float *dC, int n)
{
  __shared__ float shA[BSXY][BSXY];
  __shared__ float shB[BSXY][BSXY];
  float c = 0.0;

  int i = blockIdx.y * BSXY + threadIdx.y;
  int j = blockIdx.x * BSXY + threadIdx.x;

  if (i >= n || j >= n) {
    return;
  }

  for (int tile = 0; tile < (n + BSXY - 1) / BSXY; tile++) {
    int tiledColA = tile * BSXY + threadIdx.x;
    int tiledRowB = tile * BSXY + threadIdx.y;

    shA[threadIdx.y][threadIdx.x] = (i < n && tiledColA < n) ? dA[i * n + tiledColA] : 0.0f;
    shB[threadIdx.y][threadIdx.x] = (tiledRowB < n && j < n) ? dB[tiledRowB + j * n] : 0.0f;

    __syncthreads();

    for (int k = 0; k < BSXY; k++) {
      c += shA[threadIdx.y][k] * shB[k][threadIdx.x];
    }

    __syncthreads();
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
  cudaMalloc((void **)&dA, N * N * sizeof(float));
  cudaMalloc((void **)&dB, N * N * sizeof(float));
  cudaMalloc((void **)&dC, N * N * sizeof(float));
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);


  // Call each GPU kernel appropriately to multiply matrices A and B
  // Measure and print the execution time and performance (GFlops/s) of each kernel, without counting the data transfer time
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    dimGrid.x = (N + BSXY - 1) / BSXY;
    dimGrid.y = (N + BSXY - 1) / BSXY;
    dimGrid.z = 1;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    multiplyMatrixGPUByBlocksThreads2DNonMultipleSharedMemory<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    float gflops = (2.0f * N * N * N) / (milliseconds * 1e6);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFlops/s" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Copy the array dC back to the CPU
  cudaMemcpy(C, dC, N * N * sizeof(C[0]), cudaMemcpyDeviceToHost);
  // Verify the results
  multiplyMatrixCPU();
  verifyResults();

  // Deallocate A, B, C
  free(A); free(B); free(C);

  // Deallocate dA, dB, dC
  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  return 0;
}
