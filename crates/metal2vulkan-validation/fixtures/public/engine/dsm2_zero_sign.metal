// Owned fixture: the sign of the zero cell an 8x8 simdgroup multiply-add,
// in float and in half, reads for a lane that does not take part in it (not live in its
// block, or past the threadgroup's thread count), compiled without fast math so a signed zero
// asks a defined question (no `nsz`).
// metal-fast-math: off
//
// The kernel reads three 8x8 row-major matrices a, b, c from `in` (float, at 0, 64, 128) and from
// `hin` (half, the same values), and the live-lane mask `live[0]`. Every lane loads its fragments
// before the branch; lane L runs the multiply-add when bit L of the mask is set, a branch the
// translator cannot prove workgroup uniform. A lane that runs d = a * b + c writes its two cells:
// out[2g + j] is element j of the float d's fragment and out[2n + 2g + j] that of the half d's,
// widened, at thread g of a grid of n threads (g = thread_position_in_grid). A lane that does not
// run it writes nothing. Lane L of a simdgroup holds cells (row(L), col(L) + j),
// row(L) = ((L >> 1) & 3) | ((L >> 4) << 2), col(L) = ((L >> 3) & 1) * 4 + (L & 1) * 2.
// With a -0 accumulator and every other term -0, a cell's result is -0 exactly when every term
// is, so a term that multiplies a non-taking-part lane's zero cell by a nonzero partner shows
// that zero's sign. Loop-free.
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

kernel void dsm2_zero_sign(device const float *in [[buffer(0)]],
                           device const half *hin [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           device const uint *live [[buffer(3)]],
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
    if (((live[0] >> lane) & 1u) != 0u) {
        mad(a, b, c, ha, hb, hc, out, g, n);
    }
}
