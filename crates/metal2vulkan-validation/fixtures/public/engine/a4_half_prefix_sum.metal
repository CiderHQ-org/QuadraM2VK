// Owned fixture: `half` prefix sums over a simdgroup
// (`simd.scan.fadd.half`). One simdgroup of 32 threads sums small integers, so every partial sum
// is an integer below 2048, exact in `half` in any order of addition: out[t] is the inclusive sum
// of in[0..=t], and out[32 + t] one plus the exclusive sum (so lane 0 reads 1 whichever zero the
// empty sum is). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_half_prefix_sum(device const half *in [[buffer(0)]],
                               device half *out [[buffer(1)]],
                               uint t [[thread_index_in_threadgroup]]) {
    half x = in[t];
    out[t] = simd_prefix_inclusive_sum(x);
    out[32 + t] = simd_prefix_exclusive_sum(x) + 1.0h;
}
