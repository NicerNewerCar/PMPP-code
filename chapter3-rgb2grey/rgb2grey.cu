#include "rgb2grey.h"
__global__ void rgb2greyKernel(unsigned char *out, unsigned char *in, int m,
                               int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // Col
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

  if (i > m || j > n)
    return;

  unsigned int idx = j * m + i; // Pos in greyscale image

  unsigned char r = in[idx * 3];
  unsigned char g = in[idx * 3 + 2];
  unsigned char b = in[idx * 3 + 3];

  out[idx] = 0.21f * r + 0.72f * g + 0.07f * b;
}

void rgb2grey(unsigned char *out, unsigned char *in, int m, int n) {
  dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
  dim3 dimBlock(16, 16, 1);
  rgb2greyKernel<<<dimGrid, dimBlock>>>(out, in, m, n);
}
