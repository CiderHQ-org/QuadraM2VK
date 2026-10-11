#include <metal_stdlib>
using namespace metal;

// The two `simdgroup_load`/`simdgroup_store` operands nothing else exercises: `transpose_matrix` and
// `matrix_origin`. Both are invisible in the AIR symbol name -- the frontend encodes the transpose by
// SWAPPING the two components of the stride descriptor and of the origin together -- so a lowering
// that keys off the intrinsic name and reads a single leading dimension gets both silently wrong.
// The staging array is 16 wide so the stride is not the matrix side, and every block is read back
// through a plain store so the mapping is legible cell by cell.
kernel void kernel_simdgroup_matrix_transposed_and_offset_block(
    device const float *fa  [[buffer(0)]],
    device float       *out [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float ta[256];
    for (uint i = tid; i < 256; i += 32) ta[i] = fa[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 m0, m1, m2, m3;
    simdgroup_load(m0, ta, 16);
    simdgroup_load(m1, ta, 16, ulong2(0, 0), true);
    simdgroup_load(m2, ta, 16, ulong2(3, 2));
    simdgroup_load(m3, ta, 16, ulong2(3, 2), true);

    simdgroup_store(m0, out +   0, 8);
    simdgroup_store(m1, out +  64, 8);
    simdgroup_store(m2, out + 128, 8);
    simdgroup_store(m3, out + 192, 8);
    simdgroup_store(m0, out + 256, 16, ulong2(0, 0), true);
    simdgroup_store(m0, out + 432, 16, ulong2(3, 2));
    simdgroup_store(m0, out + 608, 16, ulong2(3, 2), true);
}
