@group(0) @binding(1) var<storage, read_write> outData : array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&outData)) {
    outData[i] = i;
  }
}
