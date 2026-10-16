// Owned fixture: an 8x8 simdgroup multiply-add, in float and in half, in a
// block the translator cannot prove workgroup uniform: simdgroup k takes the branch when the
// device word sel[k] is nonzero, a value equal across the simdgroup. Every lane of a simdgroup
// takes it or none does; the simdgroup is partial only when the threadgroup leaves it partial.
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

// (a), and (c) at a threadgroup below 32 threads: simdgroup k runs the multiply-add when
// sel[k] != 0, a device value equal across the simdgroup.
kernel void dsm_whole_by_buffer(device const float *in [[buffer(0)]],
                                device const half *hin [[buffer(1)]],
                                device float *out [[buffer(2)]],
                                device const uint *sel [[buffer(3)]],
                                uint g [[thread_position_in_grid]],
                                uint n [[threads_per_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    simdgroup_float8x8 a, b, c;
    simdgroup_half8x8 ha, hb, hc;
    simdgroup_load(a, in, 8);
    simdgroup_load(b, in + 64, 8);
    simdgroup_load(c, in + 128, 8);
    simdgroup_load(ha, hin, 8);
    simdgroup_load(hb, hin + 64, 8);
    simdgroup_load(hc, hin + 128, 8);
    if (sel[sg] != 0u) {
        mad(a, b, c, ha, hb, hc, out, g, n);
    }
}
