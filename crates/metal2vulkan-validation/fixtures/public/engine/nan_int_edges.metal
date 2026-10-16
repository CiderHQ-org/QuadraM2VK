// Owned fixture (NaN model): what Metal's conversions between float and integer
// answer at their edges. Lane `i` reads a float a, an int i, a uint u, and a half ha from `in[i]`,
// and writes eight words to `out[2i .. 2i + 1]`: int(a), uint(a), short(a) sign-extended,
// ushort(a) | uchar(a) << 16 | char(a) << 24, float(i), float(u), half(i) | half(u) << 16, and
// int(ha) (each an `air.convert`). Compiled without fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_int_edges(device const uint4 *in [[buffer(0)]],
                          device uint4 *out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x);
    int i = as_type<int>(in[tid].y);
    uint u = in[tid].z;
    half ha = as_type<half>(ushort(in[tid].w));
    out[2 * tid] = uint4(as_type<uint>(int(a)), uint(a), as_type<uint>(int(short(a))),
                         uint(ushort(a)) | uint(uchar(a)) << 16 | uint(as_type<uchar>(char(a))) << 24);
    out[2 * tid + 1] = uint4(as_type<uint>(float(i)), as_type<uint>(float(u)),
                             uint(as_type<ushort>(half(i))) | uint(as_type<ushort>(half(u))) << 16,
                             as_type<uint>(int(ha)));
}
