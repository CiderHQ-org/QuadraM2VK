// Owned fixture: an 8x8 simdgroup matrix filled with a runtime value
// (`simd.matrix_init_filled`), stored as a block and read back through `thread_elements()`. One
// simdgroup of 32 threads: out[0, 64) is the block store and out[2L + j + 64] element j of lane
// L, every one equal to fill[0]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_matrix_filled(device const float *fill [[buffer(0)]],
                             device float *out [[buffer(1)]],
                             uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 f = make_filled_simdgroup_matrix<float, 8, 8>(fill[0]);
    simdgroup_store(f, out, 8);
    out[64 + 2 * lane] = f.thread_elements()[0];
    out[64 + 2 * lane + 1] = f.thread_elements()[1];
}
