// Owned fixture: an 8x8 simdgroup multiply-add, in float and in half,
// under a branch on a simdgroup collective's result that differs by lane, after a loop whose
// exit differs by lane.
// metal-fast-math: off
//
// The kernel reads three 8x8 row-major matrices a, b, c from `in` (float, at 0, 64, 128) and from
// `hin` (half, the same values). Every lane loads its fragments, then runs a loop L + 1 times
// (L its lane in the simdgroup); iteration i's simd_sum(1) counts the lanes still in the loop,
// which is 32 - i in a whole simdgroup, so lane L leaves holding s = 32 - L. Lane L runs the
// multiply-add when s > 16: lanes 0-15 of each whole simdgroup. A lane that runs d = a * b + c
// writes its two cells: out[2g + j] is element j of the float d's fragment and out[2n + 2g + j]
// that of the half d's, widened, at thread g of a grid of n threads (g = thread_position_in_grid).
// A lane that does not run it writes nothing. Lane L of a simdgroup holds cells
// (row(L), col(L) + j), row(L) = ((L >> 1) & 3) | ((L >> 4) << 2),
// col(L) = ((L >> 3) & 1) * 4 + (L & 1) * 2.
#include <metal_stdlib>
using namespace metal;

kernel void dsm2_collective_branch(device const float *in [[buffer(0)]],
                                   device const half *hin [[buffer(1)]],
                                   device float *out [[buffer(2)]],
                                   uint g [[thread_position_in_grid]],
                                   uint n [[threads_per_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 a, b, c;
    simdgroup_half8x8 ha, hb, hc;
    simdgroup_load(a, in, 8);
    simdgroup_load(b, in + 64, 8);
    simdgroup_load(c, in + 128, 8);
    simdgroup_load(ha, hin, 8);
    simdgroup_load(hb, hin + 64, 8);
    simdgroup_load(hc, hin + 128, 8);
    uint s = 0u;
    uint k = 0u;
    do {
        s = simd_sum(1u);
        k++;
    } while (k <= lane);
    if (s > 16u) {
        simdgroup_float8x8 d;
        simdgroup_half8x8 hd;
        simdgroup_multiply_accumulate(d, a, b, c);
        simdgroup_multiply_accumulate(hd, ha, hb, hc);
        out[2u * g] = d.thread_elements()[0];
        out[2u * g + 1u] = d.thread_elements()[1];
        out[2u * n + 2u * g] = float(hd.thread_elements()[0]);
        out[2u * n + 2u * g + 1u] = float(hd.thread_elements()[1]);
    }
}
