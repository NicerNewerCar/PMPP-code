#include "rgb2grey.h"
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define QUALITY 100

inline void checkCudaError(cudaError_t err, int line_num) {
  if (err != cudaSuccess) {
    std::cout << (err) << ": " << cudaGetErrorString(err) << " in " << __FILE__
              << " at " << line_num << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *arg[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./rgb2grey <input image>" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Read in image
  int w = -1;
  int h = -1;
  int comp = -1;
  const unsigned char *img = stbi_load(arg[1], &w, &h, &comp, 3);
  size_t size = w * h * sizeof(unsigned char);

  // Create Images on GPU
  unsigned char *d_i, *d_o;
  checkCudaError(cudaMalloc((void **)&d_i, size * 3), __LINE__);
  checkCudaError(cudaMalloc((void **)&d_o, size), __LINE__);

  // Copy input image to GPU
  checkCudaError(cudaMemcpy(d_i, img, size * 3, cudaMemcpyHostToDevice),
                 __LINE__);

  // Kernel
  rgb2grey(d_o, d_i, w, h);
  checkCudaError(cudaGetLastError(), __LINE__);

  // Get Image from GPU and write it out
  unsigned char *out = (unsigned char *)malloc(size);
  checkCudaError(cudaMemcpy(out, d_o, size, cudaMemcpyDeviceToHost), __LINE__);
  stbi_write_jpg("./grey.jpg", w, h, 1, out, QUALITY);

  // Clean up
  // free(img);
  free(out);

  checkCudaError(cudaFree(d_o), __LINE__);
  checkCudaError(cudaFree(d_i), __LINE__);

  return 0;
}
