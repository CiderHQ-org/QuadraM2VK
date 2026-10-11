#include <metal_stdlib>
using namespace metal;

// A device pointer merged with `nullptr`, then INDEXED: the GEP is applied to the merge, not to
// either arm, so the emitter has to build one access chain per arm and give the null arm a
// defined value. 6 of the 14,579 corpus sources spell this and no authored case did.
//
// `has_bias` is 1 in the authored input, so Metal never dereferences the null arm -- the arm is
// present in the AIR and absent from the execution, which is exactly the state the merge is for.
kernel void gep_through_a_nullable_pointer_merge(
    device const float *bias [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &has_bias [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  device const float *p =
      has_bias != 0u ? bias + (gid & 3u) : (device const float *)nullptr;
  float v = p[gid];
  out[gid] = (p == nullptr) ? -1.0f : v;
}
