// Owned fixture: what `thread_elements()` of an 8x8 simdgroup matrix
// holds, read against the block store and the block load. One simdgroup: lane L writes
// in[2L + j] into element j of `a`; `b` is a block load of in[64, 128). The output holds b's
// elements as each lane reads them, `a` as the block store lays it out, and the product a * b
// both ways (elements and block store). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a3_matrix_thread_elements(device const float *in [[buffer(0)]],
                                      device float *out [[buffer(1)]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 a;
    a.thread_elements()[0] = in[2 * lane];
    a.thread_elements()[1] = in[2 * lane + 1];
    simdgroup_float8x8 b;
    simdgroup_load(b, in + 64, 8);
    simdgroup_float8x8 r;
    simdgroup_multiply(r, a, b);
    out[2 * lane] = b.thread_elements()[0];
    out[2 * lane + 1] = b.thread_elements()[1];
    simdgroup_store(a, out + 64, 8);
    out[128 + 2 * lane] = r.thread_elements()[0];
    out[128 + 2 * lane + 1] = r.thread_elements()[1];
    simdgroup_store(r, out + 192, 8);
}
