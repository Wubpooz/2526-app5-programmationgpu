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
  // TODO / A FAIRE ...
  __shared__ float shA[BSXY][BSXY];
  __shared__ float shB[BSXY][BSXY];
  float c = 0.0;
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
  // TODO / A FAIRE ...


  // Call each GPU kernel appropriately to multiply matrices A and B
  // Measure and print the execution time and performance (GFlops/s) of each kernel, without counting the data transfer time
  // TODO / A FAIRE ...
  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimBlock.x = 32;
    dimBlock.y = 32;
    dimBlock.z = 1;
    dimGrid.x = (N + 31) / 32;
    dimGrid.y = (N + 31) / 32;
    dimGrid.z = 1;
    // multiplyMatrixGPUByBlocksThreads2DNonMultipleSharedMemory<<<dimGrid, dimBlock>>>(N);
  }

  // Copy the array dC back to the CPU
  // TODO / A FAIRE ...

  // Verify the results
  multiplyMatrixCPU();
  verifyResults();

  // Deallocate A, B, C
  free(A); free(B); free(C);

  // Deallocate dA, dB, dC
  // TODO / A FAIRE ...

  return 0;
}
