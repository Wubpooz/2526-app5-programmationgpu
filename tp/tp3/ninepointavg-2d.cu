/**
  * In this exercise, we will implement GPU kernels for computing the average of 9 points on a 2D array.
  *
  * Kernel 1: Use 1D grid of blocks (only blockIdx.x), no additional threads (1 thread per block)
  *
  * Kernel 2: Use 2D grid of blocks (blockIdx.x/.y), no additional threads (1 thread per block)
  *
  * Kernel 3: Use 2D grid of blocks and 2D threads (BSXY x BSXY), each thread computing 1 element of Aavg
  *
  * Kernel 4: Use 2D grid of blocks and 2D threads, each thread computing 1 element of Aavg, use shared memory. Each block should load BSXY x BSXY elements of A, then compute (BSXY - 2) x (BSXY - 2) elements of Aavg. Borders of tiles loaded by different blocks must overlap to be able to compute all elements of Aavg.
  *
  * Kernel 5: Use 2D grid of blocks and 2D threads, use shared memory, each thread computes KxK elements of Aavg
  *
  * For all kernels: Make necessary memory allocations/deallocations and memcpy in the main.
*/

#include <iostream>
#include <cstdio>
#include "cuda.h"
#include "omp.h"

#define N 1024
#define K 4
#define BSXY 32

// The matrix is stored by columns, that is A(i, j) = A[i + j * N]. The average should be computed on Aavg array.
float *A;
float *Aavg;

float *dA, *dAavg;


void ninePointAverage1DGrid(float *dA, float *dAavg, int n) {
  int i = blockIdx.x + 1;
  int j = blockIdx.y + 1;

  if (i < n - 1 && j < n - 1) {
    dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + (j) * n] + dA[i - 1 + (j + 1) * n] +
        dA[i + (j - 1) * n] + dA[i + (j) * n] + dA[i + (j + 1) * n] +
        dA[i + 1 + (j - 1) * n] + dA[i + 1 + (j) * n] + dA[i + 1 + (j + 1) * n]) * (1.0 / 9.0);
  }
}
void ninePointAverage2DGrid1DBlock(float *dA, float *dAavg, int n) {}
void ninePointAverage2DGrid2DBlock(float *dA, float *dAavg, int n) {}
void ninePointAverage2DGrid2DBlockShared(float *dA, float *dAavg, int n) {}
void ninePointAverage2DGrid2DBlockSharedKElements(float *dA, float *dAavg, int n) {}



// Reference CPU implementation
void ninePointAverageCPU(const float *A, float *Aavg)
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0 / 9.0);
    }
  }
}


int main() {
  A = (float *) malloc (N * N * sizeof(float));
  Aavg = (float *) malloc (N * N * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
    }
  }

  cudaMalloc((void **)&dA, N * N * sizeof(float));
  cudaMalloc((void **)&dAavg, N * N * sizeof(float));
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  {
    dim3 dimGrid;
    dim3 dimBlock;
    dimBlock.x = BSXY;
    dimBlock.y = BSXY;
    dimBlock.z = 1;
    dimGrid.x = (N + BSXY - 1) / BSXY;
    dimGrid.y = (N + BSXY - 1) / BSXY;
    dimGrid.z = 1;

    ninePointAverage1DGrid<<<dimGrid, dimBlock>>>(dA, dAavg, N);
  }

  cudaMemcpy(Aavg, dAavg, N * N * sizeof(Aavg[0]), cudaMemcpyDeviceToHost);

  ninePointAverageCPU();
  verifyResults();

  free(A); free(Aavg);
  cudaFree(dA); cudaFree(dAavg);

  return 0;
}
