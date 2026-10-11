// Six uncovered multisample read families in one kernel.
//
// `air.read_texture_2d_ms.u.v4i16` (3 corpus sources), `air.read_texture_2d_ms.u.v4i32` (2),
// `air.read_texture_2d_ms.s.v4i32` (2), `air.read_texture_2d_ms.i16.v4f16` (2),
// `air.read_texture_2d_ms.i16.u.v4i16` (2) and `air.read_depth_2d_ms.f32` (2) -- 13 sources.
//
// A multisample read takes one operand an ordinary read does not, the sample index, and it is the
// operand a lowering can silently drop: every sample of a texel is a plausible answer for that
// texel. So the four threads read four different texels at two different samples, and every stored
// value encodes its own (x, y, sample) triple -- a read that ignored the sample index, or that took
// it from the coordinate, lands a value the derivation places somewhere else.
//
// The two `u.v4i16` symbols differ only in the coordinate width -- `.i16.` in the middle is a
// ushort2 coordinate and a ushort sample index, the other is uint2 and uint. They read the same
// texture at the same place through different channels, so a lowering that widened one wrongly
// disagrees with the other on a texture where both must be right.
//
// Every value is an integer well inside its type, or a half or float that is an exact binary
// fraction, so nothing here depends on rounding.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_multisample_reads(
    device uint *out [[buffer(0)]],
    texture2d_ms<ushort> msu16 [[texture(0)]],
    texture2d_ms<uint>   msu32 [[texture(1)]],
    texture2d_ms<int>    mss32 [[texture(2)]],
    texture2d_ms<half>   msh   [[texture(3)]],
    depth2d_ms<float>    msd   [[texture(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint2   c32 = uint2(tid % 2, tid / 2);
    ushort2 c16 = ushort2(ushort(tid % 2), ushort(tid / 2));
    uint    s32 = tid % 2;
    ushort  s16 = ushort(tid % 2);

    ushort4 a = msu16.read(c32, s32);
    uint4   b = msu32.read(c32, s32);
    int4    c = mss32.read(c32, s32);
    half4   d = msh.read(c16, s16);
    ushort4 e = msu16.read(c16, s16);
    float   f = msd.read(c32, s32);

    out[tid * 6 + 0] = uint(a.x);
    out[tid * 6 + 1] = b.y;
    out[tid * 6 + 2] = uint(c.z);
    out[tid * 6 + 3] = uint(as_type<ushort>(d.w));
    out[tid * 6 + 4] = uint(e.y);
    out[tid * 6 + 5] = as_type<uint>(f);
}
