extern "C" __global__ void kernel(unsigned int* out) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = i;
}
