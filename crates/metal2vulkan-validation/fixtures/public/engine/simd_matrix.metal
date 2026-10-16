#include <metal_stdlib>
using namespace metal;

// Simdgroup matrices: element access through `thread_elements()`, filled and
// diagonal matrices, products with mixed element formats, and block loads and stores at a
// leading dimension and origin. One simdgroup of 32 threads per threadgroup; loop-free.

// Each lane writes its two elements of `a` from `in`, multiplies by the diagonal 2 and adds the
// filled 0.5, then writes its two result elements back and stores the whole product.
kernel void matrix_thread_elements(
    device const float *in  [[buffer(0)]],
    device float       *out [[buffer(1)]],
    device float       *blk [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]])
{
    simdgroup_float8x8 a;
    a.thread_elements()[0] = in[2 * lane];
    a.thread_elements()[1] = in[2 * lane + 1];
    simdgroup_float8x8 d = simdgroup_float8x8(2.0f);
    simdgroup_float8x8 f = make_filled_simdgroup_matrix<float, 8, 8>(0.5f);
    simdgroup_float8x8 r;
    simdgroup_multiply_accumulate(r, a, d, f);
    out[2 * lane]     = r.thread_elements()[0];
    out[2 * lane + 1] = r.thread_elements()[1];
    simdgroup_store(r, blk, 8);
}

// Products of loaded matrices in every mix of `half` and `float` the frontend spells, and a
// `bfloat` product accumulated in `float`.
kernel void matrix_mixed(
    device const float  *fa  [[buffer(0)]],
    device const half   *ha  [[buffer(1)]],
    device const bfloat *ba  [[buffer(2)]],
    device float        *fo  [[buffer(3)]],
    device half         *ho  [[buffer(4)]])
{
    simdgroup_float8x8  f, fr1, fr2, fr3;
    simdgroup_half8x8   h, hr1, hr2;
    simdgroup_bfloat8x8 b;
    simdgroup_load(f, fa, 8);
    simdgroup_load(h, ha, 8);
    simdgroup_load(b, ba, 8);
    simdgroup_multiply_accumulate(fr1, f, h, f);
    simdgroup_multiply_accumulate(fr2, h, h, f);
    simdgroup_multiply_accumulate(fr3, b, b, f);
    simdgroup_multiply_accumulate(hr1, h, h, h);
    simdgroup_multiply_accumulate(hr2, f, h, h);
    simdgroup_store(fr1, fo, 8);
    simdgroup_store(fr2, fo + 64, 8);
    simdgroup_store(fr3, fo + 128, 8);
    simdgroup_store(hr1, ho, 8);
    simdgroup_store(hr2, ho + 64, 8);
}

// A product of two blocks of a 16-wide threadgroup array, one transposed and at an origin,
// stored transposed at another origin.
kernel void matrix_threadgroup_block(
    device const float *in  [[buffer(0)]],
    device float       *out [[buffer(1)]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float t[256];
    t[lane]       = in[lane];
    t[lane + 32]  = in[lane + 32];
    t[lane + 64]  = in[lane + 64];
    t[lane + 96]  = in[lane + 96];
    t[lane + 128] = in[lane + 128];
    t[lane + 160] = in[lane + 160];
    t[lane + 192] = in[lane + 192];
    t[lane + 224] = in[lane + 224];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 a, b, r;
    simdgroup_load(a, t, 16);
    simdgroup_load(b, t, 16, ulong2(3, 2), true);
    simdgroup_multiply(r, a, b);
    simdgroup_store(r, out, 16, ulong2(5, 1), true);
}
