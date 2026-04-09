#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
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
  // TODO / A FAIRE ...
}

int main()
{
  srand(1234);
  int N = 100000000;
  int numBlocks;// = ???; (TODO / A FAIRE ...)
  float *A, *dA; // Le tableau dont minimum on va chercher
  float *Amin, *dAmin; // Amin contiendra en suite le tableau reduit par un facteur de BLOCKSIZE apres l'execution du kernel GPU

  // Allocate arrays A[N] and Amin[numBlocks] on the CPU
  // Allour les tableaux A[N] et Amin[numBlocks] sur le CPU
  // TODO / A FAIRE ...

  // Initialize the array A, set the minimum to -1
  for (int i = 0; i < N; i++) { A[i] = (float)(rand() % 1000); }
  A[rand() % N] = -1.0; 

  // Transfer A on the GPU (dA) with cudaMemcpy
  // TODO / A FAIRE ...

  // Put maximum attainable value to minA.
  float minA = FLT_MAX; 

  // Find the minimum of the array dA for each thread block, put it in dAMin[...] and transfer to the CPU, then find the global minimum of this smaller array and put it in minA.
  // TODO / A FAIRE ...
  // findMinimum<<<...>>>(...)
  // ...

  // Verify the result
  if (minA == -1) { cout << "The minimum is correct!" << endl; }
  else { cout << "The minimum found (" << minA << ") is incorrect (it should have been -1)!" << endl; }

  return 0;
}
