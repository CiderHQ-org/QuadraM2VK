// Owned fixture: an 8x8 simdgroup multiply-add, in float and in half, run by
// lanes 0-15 of each simdgroup only, which load the fragments inside the branch: lanes 16-31
// never load theirs.
//
// The kernel reads three 8x8 row-major matrices a, b, c from `in` (float, at 0, 64, 128) and from
// `hin` (half, the same values), computes d = a * b + c in both formats, and has each lane that
// runs the multiply-add write its two cells: out[2g + j] is element j of the float d's fragment
// and out[2n + 2g + j] that of the half d's, widened, at thread g of a grid of n threads
// (g = thread_position_in_grid). A lane that does not run it writes nothing. Lane L of a
// simdgroup holds cells (row(L), col(L) + j), row(L) = ((L >> 1) & 3) | ((L >> 4) << 2),
// col(L) = ((L >> 3) & 1) * 4 + (L & 1) * 2. Loop-free.
#include <metal_stdlib>
using namespace metal;

// d = a * b + c in both formats, each lane writing its two cells.
static void mad(thread simdgroup_float8x8 &a, thread simdgroup_float8x8 &b,
                thread simdgroup_float8x8 &c, thread simdgroup_half8x8 &ha,
                thread simdgroup_half8x8 &hb, thread simdgroup_half8x8 &hc,
                device float *out, uint g, uint n) {
    simdgroup_float8x8 d;
    simdgroup_half8x8 hd;
    simdgroup_multiply_accumulate(d, a, b, c);
    simdgroup_multiply_accumulate(hd, ha, hb, hc);
    out[2u * g] = d.thread_elements()[0];
    out[2u * g + 1u] = d.thread_elements()[1];
    out[2u * n + 2u * g] = float(hd.thread_elements()[0]);
    out[2u * n + 2u * g + 1u] = float(hd.thread_elements()[1]);
}

// (b) Lanes 0-15 load the fragments and run the multiply-add inside the branch: lanes 16-31
// never load theirs.
kernel void dsm_half_live_inside(device const float *in [[buffer(0)]],
                                 device const half *hin [[buffer(1)]],
                                 device float *out [[buffer(2)]],
                                 uint g [[thread_position_in_grid]],
                                 uint n [[threads_per_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    if (lane < 16u) {
        simdgroup_float8x8 a, b, c;
        simdgroup_half8x8 ha, hb, hc;
        simdgroup_load(a, in, 8);
        simdgroup_load(b, in + 64, 8);
        simdgroup_load(c, in + 128, 8);
        simdgroup_load(ha, hin, 8);
        simdgroup_load(hb, hin + 64, 8);
        simdgroup_load(hc, hin + 128, 8);
        mad(a, b, c, ha, hb, hc, out, g, n);
    }
}
