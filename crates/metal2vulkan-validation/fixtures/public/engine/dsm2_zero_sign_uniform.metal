// Owned fixture: dsm2_zero_sign in workgroup-uniform control flow (every
// thread of the threadgroup runs the multiply-add), compiled without fast math. With a
// threadgroup whose size is not a multiple of 32 the last simdgroup is partial, and its lanes
// past the thread count do not exist: their cells are zero, and this kernel shows
// that zero's sign.
// metal-fast-math: off
//
// The kernel reads three 8x8 row-major matrices a, b, c from `in` (float, at 0, 64, 128) and from
// `hin` (half, the same values), computes d = a * b + c in both formats, and has each lane write
// its two cells: out[2g + j] is element j of the float d's fragment and out[2n + 2g + j] that of
// the half d's, widened, at thread g of a grid of n threads (g = thread_position_in_grid). Lane L
// of a simdgroup holds cells (row(L), col(L) + j), row(L) = ((L >> 1) & 3) | ((L >> 4) << 2),
// col(L) = ((L >> 3) & 1) * 4 + (L & 1) * 2. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void dsm2_zero_sign_uniform(device const float *in [[buffer(0)]],
                                   device const half *hin [[buffer(1)]],
                                   device float *out [[buffer(2)]],
                                   uint g [[thread_position_in_grid]],
                                   uint n [[threads_per_grid]]) {
    simdgroup_float8x8 a, b, c, d;
    simdgroup_half8x8 ha, hb, hc, hd;
    simdgroup_load(a, in, 8);
    simdgroup_load(b, in + 64, 8);
    simdgroup_load(c, in + 128, 8);
    simdgroup_load(ha, hin, 8);
    simdgroup_load(hb, hin + 64, 8);
    simdgroup_load(hc, hin + 128, 8);
    simdgroup_multiply_accumulate(d, a, b, c);
    simdgroup_multiply_accumulate(hd, ha, hb, hc);
    out[2u * g] = d.thread_elements()[0];
    out[2u * g + 1u] = d.thread_elements()[1];
    out[2u * n + 2u * g] = float(hd.thread_elements()[0]);
    out[2u * n + 2u * g + 1u] = float(hd.thread_elements()[1]);
}
