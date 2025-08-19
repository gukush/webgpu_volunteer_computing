__kernel void kernel(__global uint* out) {
  uint i = get_global_id(0);
  out[i] = i;
}
