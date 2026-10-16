// Owned fixture: an 8x8 simdgroup multiply-add, in float and in half, run by
// lanes 0-15 of each simdgroup only, on fragments the source builds in every lane before the
// branch from plain loads (the front end moves them into the branch).
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

// The fragments of a, b, c as each lane's two cells, read by plain loads.
static void cells(device const float *in, device const half *hin, uint lane,
                  thread simdgroup_float8x8 &a, thread simdgroup_float8x8 &b,
                  thread simdgroup_float8x8 &c, thread simdgroup_half8x8 &ha,
                  thread simdgroup_half8x8 &hb, thread simdgroup_half8x8 &hc) {
    uint row = ((lane >> 1) & 3u) | ((lane >> 4) << 2);
    uint col = ((lane >> 3) & 1u) * 4u + (lane & 1u) * 2u;
    uint at = row * 8u + col;
    a.thread_elements()[0] = in[at];
    a.thread_elements()[1] = in[at + 1u];
    b.thread_elements()[0] = in[64u + at];
    b.thread_elements()[1] = in[64u + at + 1u];
    c.thread_elements()[0] = in[128u + at];
    c.thread_elements()[1] = in[128u + at + 1u];
    ha.thread_elements()[0] = hin[at];
    ha.thread_elements()[1] = hin[at + 1u];
    hb.thread_elements()[0] = hin[64u + at];
    hb.thread_elements()[1] = hin[64u + at + 1u];
    hc.thread_elements()[0] = hin[128u + at];
    hc.thread_elements()[1] = hin[128u + at + 1u];
}

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

// (b) Lanes 0-15 run the multiply-add on fragments every lane built before the branch from
// plain loads, which the compiler may move into the branch.
kernel void dsm_half_live_built(device const float *in [[buffer(0)]],
                                device const half *hin [[buffer(1)]],
                                device float *out [[buffer(2)]],
                                uint g [[thread_position_in_grid]],
                                uint n [[threads_per_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 a, b, c;
    simdgroup_half8x8 ha, hb, hc;
    cells(in, hin, lane, a, b, c, ha, hb, hc);
    if (lane < 16u) {
        mad(a, b, c, ha, hb, hc, out, g, n);
    }
}
