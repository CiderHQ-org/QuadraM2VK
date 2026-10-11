#include <metal_stdlib>
using namespace metal;

// `simdgroup_load`/`simdgroup_store` against THREADGROUP memory (`air.*.p3f32`/`p3f16`), the
// address space the device-pointer fixture `kernel_simdgroup_matrix_block` does not reach. The
// staging arrays are wider than the 8x8 block so the leading dimension is 16, not 8: the element
// address is `row * elements_per_row + col`, and an 8-wide stride cannot tell the two apart.
kernel void kernel_simdgroup_matrix_threadgroup_block(
    device const float *fa  [[buffer(0)]],
    device const float *fb  [[buffer(1)]],
    device const half  *ha  [[buffer(2)]],
    device uint        *out [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float ta[128];
    threadgroup float tb[128];
    threadgroup half  th[128];
    threadgroup float ts[128];
    threadgroup half  ths[128];

    for (uint i = tid; i < 128; i += 32) {
        ta[i]  = fa[i];
        tb[i]  = fb[i];
        th[i]  = ha[i];
        ts[i]  = 0.0f;
        ths[i] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 a, b, r;
    simdgroup_half8x8  h, rh;
    simdgroup_load(a, ta, 16);
    simdgroup_load(b, tb, 16);
    simdgroup_load(h, th, 16);

    simdgroup_multiply_accumulate(r, a, b, a);
    simdgroup_multiply_accumulate(rh, h, simdgroup_half8x8(2.0h), h);

    simdgroup_store(r, ts, 16);
    simdgroup_store(rh, ths, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 128; i += 32) {
        out[i]       = as_type<uint>(ts[i]);
        out[i + 128] = uint(as_type<ushort>(ths[i]));
    }
}
