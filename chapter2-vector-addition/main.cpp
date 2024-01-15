#include "vecAdd.h"
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

inline void checkCudaError(cudaError_t err, int line_num) {
  if (err != cudaSuccess) {
    std::cout << (err) << ": " << cudaGetErrorString(err) << " in " << __FILE__
              << " at " << line_num << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  // CUDA Vector Add Function
  cudaError_t err;

  // Init some C arrays
  int N = 10000;
  size_t array_size = N * sizeof(float);
  float *a = (float *)malloc(array_size);
  float *b = (float *)malloc(array_size);
  float *c = (float *)malloc(array_size);
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    c[i] = 0.0;
  }

  // Allocate memory on the GPU
  float *d_a, *d_b, *d_c;
  checkCudaError(cudaMalloc((void **)&d_a, array_size), __LINE__);
  checkCudaError(cudaMalloc((void **)&d_b, array_size), __LINE__);
  checkCudaError(cudaMalloc((void **)&d_c, array_size), __LINE__);

  // Copy A and B to GPU
  checkCudaError(cudaMemcpy(d_a, a, array_size, cudaMemcpyHostToDevice),
                 __LINE__);
  checkCudaError(cudaMemcpy(d_b, b, array_size, cudaMemcpyHostToDevice),
                 __LINE__);

  // Kernel Code
  vecAdd(d_a, d_b, d_c, N);

  // Copy result into C
  checkCudaError(cudaMemcpy(c, d_c, array_size, cudaMemcpyDeviceToHost),
                 __LINE__);

  // Print Results
  for (int i = 0; i < N; i++) {
    std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
  }

  // Free C arrays
  free(a);
  free(b);
  free(c);

  // Free GPU memory
  checkCudaError(cudaFree(d_a), __LINE__);
  checkCudaError(cudaFree(d_b), __LINE__);
  checkCudaError(cudaFree(d_c), __LINE__);
  return 0;
}
