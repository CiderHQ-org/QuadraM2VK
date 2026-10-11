// Thirteen uncovered subgroup, quad and null-texture families in one 32-thread kernel.
//
// `air.sincos.f16` (6 corpus sources), `air.simd_max.f16` (3), `air.simd_broadcast.s.i16` (3),
// `air.quad_sum.v3f16` (3), `air.get_null_texture_3d` (3), `air.simd_min.f16` (2),
// `air.simd_min.s.i16` (2), `air.simd_or.u.i8` (2), `air.simd_any` (2),
// `air.simd_prefix_exclusive_sum.s.i32` (2), `air.quad_sum.v2f16` (2), `air.quad_sum.f16` (2) and
// `air.sign.v4f16` (2) -- 34 sources.
//
// The dispatch is exactly one 32-thread threadgroup, which is one SIMD group on this hardware. That
// is what makes a subgroup reduction a derivable quantity rather than a hardware-shaped one: every
// lane is known, so every reduction has one answer.
//
// Each family is asked something only it can answer:
//
//  - The reductions see a value that is distinct per lane and whose extreme sits at a known lane --
//    `simd_max` at lane 31 and `simd_min` at lane 0 over `tid / 4`, `simd_min` over `tid - 10` at
//    lane 0 again but negative, so a lowering that ran the unsigned comparison lands elsewhere.
//  - `simd_or` sees one distinct bit per lane modulo eight, so all eight bits are set only if every
//    lane contributed; dropping any lane leaves a hole.
//  - `simd_prefix_exclusive_sum` is `i * (i - 1) / 2` at lane i, which is different at every lane
//    and different from the inclusive sum at all of them -- the one distinction that matters here.
//  - `quad_sum` is asked at three widths. A quad of a 1D dispatch is four consecutive lanes, so the
//    answer is `16q + 6` scaled per component, which differs per quad; a reduction that spanned the
//    whole SIMD group instead lands one value for all 32 lanes.
//  - `sign` sees `x - 3.875`, which no lane makes zero (`tid / 4` is a multiple of a quarter), so
//    the answer flips from -1 to +1 at lane 16 and no signed zero is involved.
//  - `is_null_texture` is asked of both a default-constructed `texture3d` and the bound one, so it
//    must answer both ways in the same kernel. `air.get_null_texture_3d` is the null one.
//
// `sincos` is the only inexact family. Metal and MoltenVK agree on half sin and cos, so the case is
// evidence about the lowering rather than the runtime; lane 0 sees zero, where the answers are
// exactly 0 and 1, and the rest are checked against a float64 derivation rounded to binary16.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_simd_quad_and_null_texture(
    device uint *out [[buffer(0)]],
    device const uint *in [[buffer(1)]],
    texture3d<float> bound [[texture(0)]],
    uint tid [[thread_position_in_grid]])
{
    half   x  = half(as_type<float>(in[tid]));
    half   c;
    half   s  = sincos(x, c);
    half   mx = simd_max(x);
    half   mn = simd_min(x);
    short  si = short(tid) - 10;
    short  smn = simd_min(si);
    short  bc = simd_broadcast(si, 0);
    uchar  ob = simd_or(uchar(1u << (tid % 8)));
    bool   any = simd_any(tid == 7);
    int    pre = simd_prefix_exclusive_sum(int(tid));
    half3  q3 = quad_sum(half3(half(tid), half(tid) * 2.0h, half(tid) * 4.0h));
    half2  q2 = quad_sum(half2(half(tid), half(tid) * 8.0h));
    half   q1 = quad_sum(half(tid));
    half4  sg = sign(half4(x - 3.875h, 3.875h - x, 2.0h, -2.0h));
    texture3d<float> nullTex = texture3d<float>();

    uint b = tid * 16;
    out[b +  0] = uint(as_type<ushort>(s));
    out[b +  1] = uint(as_type<ushort>(c));
    out[b +  2] = uint(as_type<ushort>(mx));
    out[b +  3] = uint(as_type<ushort>(mn));
    out[b +  4] = uint(int(smn));
    out[b +  5] = uint(int(bc));
    out[b +  6] = uint(ob);
    out[b +  7] = uint(any);
    out[b +  8] = uint(pre);
    out[b +  9] = uint(as_type<ushort>(q3.z));
    out[b + 10] = uint(as_type<ushort>(q2.y));
    out[b + 11] = uint(as_type<ushort>(q1));
    out[b + 12] = uint(as_type<ushort>(sg.x));
    out[b + 13] = uint(as_type<ushort>(sg.y));
    out[b + 14] = uint(is_null_texture(nullTex));
    out[b + 15] = uint(is_null_texture(bound));
}
