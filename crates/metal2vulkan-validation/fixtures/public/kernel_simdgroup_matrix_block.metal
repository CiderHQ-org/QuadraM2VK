#include <metal_stdlib>
using namespace metal;

kernel void kernel_simdgroup_matrix_block(
    device const float *fa  [[buffer(0)]],
    device const float *fb  [[buffer(1)]],
    device const float *fc  [[buffer(2)]],
    device const half  *ha  [[buffer(3)]],
    device const half  *hc  [[buffer(4)]],
    device uint        *out [[buffer(5)]],
    device float       *fs  [[buffer(6)]],
    device half        *hs  [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]])
{
    simdgroup_float8x8 a, b, c, r1, r2, r3;
    simdgroup_half8x8  ah, ch, rh1, rh2;
    simdgroup_load(a, fa, 8);
    simdgroup_load(b, fb, 8);
    simdgroup_load(c, fc, 8);
    simdgroup_load(ah, ha, 8);
    simdgroup_load(ch, hc, 8);

    simdgroup_float8x8 fi = simdgroup_float8x8(1.0f);
    simdgroup_half8x8  hi = simdgroup_half8x8(2.0h);

    simdgroup_multiply_accumulate(r1, a, b, c);
    simdgroup_multiply_accumulate(r2, ah, fi, c);
    simdgroup_multiply_accumulate(r3, a, fi, c);
    simdgroup_multiply_accumulate(rh1, ah, hi, ch);
    simdgroup_multiply_accumulate(rh2, a, hi, c);

    simdgroup_store(r1, fs, 8);
    simdgroup_store(r2, fs + 64, 8);
    simdgroup_store(r3, fs + 128, 8);
    simdgroup_store(rh1, hs, 8);
    simdgroup_store(rh2, hs + 64, 8);
    threadgroup_barrier(mem_flags::mem_device);

    out[tid]       = as_type<uint>(fs[tid]);
    out[tid +  32] = as_type<uint>(fs[tid +  32]);
    out[tid +  64] = as_type<uint>(fs[tid +  64]);
    out[tid +  96] = as_type<uint>(fs[tid +  96]);
    out[tid + 128] = as_type<uint>(fs[tid + 128]);
    out[tid + 160] = as_type<uint>(fs[tid + 160]);
    out[tid + 192] = uint(as_type<ushort>(hs[tid]));
    out[tid + 224] = uint(as_type<ushort>(hs[tid +  32]));
    out[tid + 256] = uint(as_type<ushort>(hs[tid +  64]));
    out[tid + 288] = uint(as_type<ushort>(hs[tid +  96]));
}
