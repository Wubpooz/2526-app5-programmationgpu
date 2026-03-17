#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

__global__ void cudaCopyByBlocks(float *tab0, const float *tab1, int size)
{
  int idx;
  // Compute the correct idx
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

__global__ void cudaCopyByBlocksThreads(float *tab0, const float *tab1, int size)
{
  int idx;
  // Compute the correct idx in terms of blockIdx.x, threadIdx.x, and blockDim.x
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

int main(int argc, char **argv) {
  float *A, *B, *dA, *dB;
  int N, i;

  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 0;
  }
  N = atoi(argv[1]);

  // Initialization
  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) { 
    A[i] = (float)i;
    B[i] = 0.0f;
  }
  
  // Allocate dynamic arrays dA and dB of size N on the GPU with cudaMalloc
  cudaError_t cudaerr = cudaMalloc((void**)&dA, sizeof(float) * N);
  if (cudaerr != cudaSuccess) {
    printf("cudaMalloc failed for dA with error: \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
  }
  cudaerr = cudaMalloc((void**)&dB, sizeof(float) * N);
  if (cudaerr != cudaSuccess) {
    printf("cudaMalloc failed for dB with error: \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
  }

  // Copy A into dA and B into dB
  cudaerr = cudaMemcpy(dA, A, sizeof(float) * N, cudaMemcpyHostToDevice);
  if (cudaerr != cudaSuccess) {
    printf("cudaMemcpy failed for dA with error: \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
  }
  cudaerr = cudaMemcpy(dB, B, sizeof(float) * N, cudaMemcpyHostToDevice);
  if (cudaerr != cudaSuccess) {
    printf("cudaMemcpy failed for dB with error: \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
  }

  // Copy dA into dB using the kernel cudaCopyByBlocks
  cudaCopyByBlocks<<<1, N>>>(dB, dA, N);

  // Wait for kernel cudaCopyByBlocks to finish
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  cudaerr = cudaMemcpy(B, dB, sizeof(float) * N, cudaMemcpyDeviceToHost);
  if (cudaerr != cudaSuccess) {
    printf("cudaMemcpy failed for B with error: \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
  }

  // Verify the results on the CPU by comparing B with A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Reinitialize B to zero, then copy B into dB again to test the second copy kernel
  for (int i = 0; i < N; i++) { B[i] = 0.0f; }
  cudaerr = cudaMemcpy(dB, B, sizeof(float) * N, cudaMemcpyHostToDevice);
  if (cudaerr != cudaSuccess) {
    printf("cudaMemcpy failed for dB with error: \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
  }

  // Copy dA into dB with the kernel cudaCopyByBlocksThreads
  cudaCopyByBlocksThreads<<<(N + 255) / 256, 256>>>(dB, dA, N);

  // Wait for the kernel cudaCopyByBlocksThreads to finish
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  cudaerr = cudaMemcpy(B, dB, sizeof(float) * N, cudaMemcpyDeviceToHost);
  if (cudaerr != cudaSuccess) {
    printf("cudaMemcpy failed for B with error: \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
  }

  // Verify the results on the CPU by comparing B with A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }

  // Deallocate arrays dA[N] and dB[N] on the GPU
  cudaFree(dA);
  cudaFree(dB);

  // Deallocate A and B
  free(A);
  free(B);

  return 0;
}
