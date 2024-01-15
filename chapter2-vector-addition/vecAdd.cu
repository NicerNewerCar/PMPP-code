#include "vecAdd.h"
#include <cuda_runtime_api.h>
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

__host__ void vecAdd(float *A, float *B, float *C, int N) {
  vecAddKernel<<<ceil(N / 256.0), 256>>>(A, B, C, N);
  cudaDeviceSynchronize();
}
