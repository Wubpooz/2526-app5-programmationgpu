#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

#define N 1024

// Define an static array dA[N] of floats on the GPU
__device__ float dA[N];

int main() {
  float A[N], B[N];
  int i;

  // Initialization
  for (i = 0; i < N; i++) { A[i] = (float)i; }

  // cudaMemcpy from A[N] to dA[N]
  cudaMemcpyToSymbol(dA, A, N * sizeof(float));

  // cudaMemcpy from dA[N} to B[N]
  cudaMemcpyFromSymbol(B, dA, N * sizeof(float));

  // Wait for GPU kernels to terminate
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verify the results
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "The copy is incorrect!\n"; }
  else { cout << "The copy is correct!\n"; }

  return 0;
}
