__device__ int BLUR_SIZE = 7;

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w,
                           int h) {
  // Calculate the row and column
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Check if we are out of bounds with the image.
  if (col > w || row > h)
    return;

  int sum_values = 0;
  int num_pixels = 0;
  // Get the average of the BLUR_SIZE x BLUR_SIZE block
  for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; ++blur_row) {
    for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; ++blur_col) {
      int cur_row = row + blur_row;
      int cur_col = col + blur_col;

      if (cur_row < 0 || cur_row > h)
        continue;
      if (cur_col < 0 || cur_col > w)
        continue;

      sum_values += in[cur_row * w + cur_col];
      num_pixels++;
    }
  }

  out[row * w + col] = (unsigned char)(sum_values / num_pixels);
}

void blur(unsigned char *in, unsigned char *out, int w, int h) {
  dim3 dimGrid(ceil(w / 16.0), ceil(h / 16.0), 1);
  dim3 dimBlock(16, 16, 1);
  blurKernel<<<dimGrid, dimBlock>>>(in, out, w, h);
}
