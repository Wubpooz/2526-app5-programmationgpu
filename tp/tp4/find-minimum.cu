#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <algorithm>
#include "cuda.h"
#include <cfloat>

#define BLOCKSIZE 1024

using namespace std;


/**
  * Step 1: Write a 1D (blocks and threads) GPU kernel that finds the minimum element in an array dA[N] in each block, then writes the minimum in dAmin[blockIdx.x]. CPU should take this array and find the global minimum by iterating over this array.
  * Step 2: The first call to findMinimum reduces the size of the array to N/BLOCKSIZE. In this version, use findMinimum a second time on this resulting array, in order to reduce the size to N/(BLOCKSIZE*BLOCKSIZE) so that computation on the CPU to find the global minimum becomes negligible.
  * To find the minimum of two floats on a GPU, use the function fminf(x, y).
  */
__global__ void findMinimum(float *dA, float *dAmin, int N)
{
  __shared__ volatile float buff[BLOCKSIZE];
  int idx = threadIdx.x + blockIdx.x * BLOCKSIZE;
  if (idx < N) {
    buff[threadIdx.x] = dA[idx];
  }
  else {
    buff[threadIdx.x] = FLT_MAX;
  } // If the thread is out of bounds, set its value to the maximum possible value so it won't affect the minimum

  // Reduction in shared memory
  for (int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (threadIdx.x < s) {
      buff[threadIdx.x] = fminf(buff[threadIdx.x], buff[threadIdx.x + s]);
    }
  }

  // Write the minimum of this block to dAmin
  if (threadIdx.x == 0) {
    dAmin[blockIdx.x] = buff[0];
  }
}

template <int BLOCK_SIZE>
__global__ void findMinimumOptimized(float *dA, float *dAmin, int N)
{
  __shared__ volatile float buff[BLOCK_SIZE > 64 ? BLOCK_SIZE : 64];
  int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  int tid = threadIdx.x;

  if (idx < N) {
    buff[tid] = dA[idx];
  } else {
    buff[tid] = FLT_MAX;
  }
  __syncthreads();

  if (BLOCK_SIZE > 512 && tid < 512) {
    buff[tid] = fminf(buff[tid], buff[tid + 512]);
  }
  __syncthreads();

  if (BLOCK_SIZE > 256 && tid < 256) {
    buff[tid] = fminf(buff[tid], buff[tid + 256]);
  }
  __syncthreads();

  if (BLOCK_SIZE > 128 && tid < 128) {
    buff[tid] = fminf(buff[tid], buff[tid + 128]);
  }
  __syncthreads();

  if (BLOCK_SIZE > 64 && tid < 64) {
    buff[tid] = fminf(buff[tid], buff[tid + 64]);
  }
  __syncthreads();

  if (tid < 32) {
    if (BLOCK_SIZE > 32) {
      buff[tid] = fminf(buff[tid], buff[tid + 32]);
    }
    if (BLOCK_SIZE > 16) {
      buff[tid] = fminf(buff[tid], buff[tid + 16]);
    }
    if (BLOCK_SIZE > 8) {
      buff[tid] = fminf(buff[tid], buff[tid + 8]);
    }
    if (BLOCK_SIZE > 4) {
      buff[tid] = fminf(buff[tid], buff[tid + 4]);
    }
    if (BLOCK_SIZE > 2) {
      buff[tid] = fminf(buff[tid], buff[tid + 2]);
    }
    if (BLOCK_SIZE > 1) {
      buff[tid] = fminf(buff[tid], buff[tid + 1]);
    }
  }

  if (tid == 0) {
    dAmin[blockIdx.x] = buff[0];
  }
}

int main()
{
  srand(1234);
  int N = BLOCKSIZE * BLOCKSIZE;
  int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
  float *A, *dA; // Le tableau dont minimum on va chercher
  float *Amin, *dAmin; // Amin contiendra en suite le tableau reduit par un facteur de BLOCKSIZE apres l'execution du kernel GPU

  // Allocate arrays A[N] and Amin[numBlocks] on the CPU
  A = (float *) malloc(N * sizeof(float));
  Amin = (float *) malloc(numBlocks * sizeof(float));

  // Initialize the array A, set the minimum to -1
  for (int i = 0; i < N; i++) { A[i] = (float)(rand() % 1000); }
  A[rand() % N] = -1.0; 

  // Transfer A on the GPU (dA) with cudaMemcpy
  cudaMalloc(&dA, N * sizeof(float));
  cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&dAmin, numBlocks * sizeof(float));
  cudaMemcpy(dAmin, Amin, numBlocks * sizeof(float), cudaMemcpyHostToDevice);

  // Put maximum attainable value to minA.
  float minA = FLT_MAX; 

  // Find the minimum of the array dA for each thread block, put it in dAMin[...] and transfer to the CPU, then find the global minimum of this smaller array and put it in minA.
  findMinimum<<<numBlocks, BLOCKSIZE>>>(dA, dAmin, N);
  findMinimum<<<1, BLOCKSIZE>>>(dAmin, dAmin, numBlocks); // On peut faire un second appel pour reduire la taille de l'array a N/(BLOCKSIZE*BLOCKSIZE)
  cudaMemcpy(Amin, dAmin, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

  // Final CPU reduction
  minA = *std::min_element(Amin, Amin + numBlocks);

  // Verify the result
  if (minA == -1) { cout << "The minimum is correct!" << endl; }
  else { cout << "The minimum found (" << minA << ") is incorrect (it should have been -1)!" << endl; }

  free(A); free(Amin);
  cudaFree(dA); cudaFree(dAmin);

  return 0;
}
