#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

int main(int argc, char **argv) {
  float *A, *B, *dA;
  int N, i;

  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 0;
  }
  N = atoi(argv[1]);

  // Initialization
  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) { A[i] = (float)i; }
  
  // Allocate the dynamic float array dA[N] on the GPU using cudaMalloc
  cudaMalloc((void **) &dA, sizeof(float) * N);

  // cudaMemcpy from A[N] to dA[N]
  cudaMemcpy(dA, A, sizeof(float) * N, cudaMemcpyHostToDevice);
 
  // cudaMemcpy from dA[N} to B[N]
  cudaMemcpy(B, dA, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Desaollouer le tableau dA[N] sur le GPU
  cudaFree(dA);

  // Attendre que les kernels GPUs terminent
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Verify the result
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "La copie est incorrecte!\n"; }
  else { cout << "La copie est correcte!\n"; }
  free(A);
  free(B);

  return 0;
}
