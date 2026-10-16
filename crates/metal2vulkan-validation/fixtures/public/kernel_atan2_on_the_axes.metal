// `air.atan2.f16` (6 corpus sources) and `air.atan2.f32` (1) had no authored case, and they are the
// last two members of the arctangent family that could get one. `air.fast_atan2.f32` cannot: at
// float width SPIRV-Cross renders the GLSL Atan2 as `precise::atan2`, and `fast::atan2` and
// `precise::atan2` disagree on a third of ordinary arguments, so a fast_ case would be asserting a
// MoltenVK limitation rather than a translation.
//
// The rows ask only C99 Annex F's special operands of atan2: the axes, the infinities, and both
// signed zeros. The claims are the signed-zero rows: atan2(-0, +1) is -0, not +0, and
// atan2(-0, -1) is -pi, not +pi. Every row answers +-0, +-pi, +-pi/2 or pi/4.
//
// Compiled with -fno-fast-math. Under the default fast-math compile every call
// carries `nnan ninf nsz`, so an infinite operand is poison and either zero is a correct result:
// those rows asked nothing Metal defines. Without the flags they are defined, and
// the symbols are the same `air.atan2.f16` and `air.atan2.f32`.
//
// The origin is deliberately ABSENT. `fast::atan2(0, 0)` is NaN where `precise::atan2(0, 0)` is 0,
// so it is the one argument in this family that discriminates between the two Metal variants -- and
// a case that asks it would be recording which variant a runtime happened to reach.
//
// Every argument is read from a buffer so nothing folds, and results are bit-cast into one `uint`
// buffer with halves zero-extended.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_atan2_on_the_axes(
    device const half* hy [[buffer(0)]],
    device const half* hx [[buffer(1)]],
    device const float* fy [[buffer(2)]],
    device const float* fx [[buffer(3)]],
    device uint* out [[buffer(4)]])
{
    // air.atan2.f16 -- the ten directions along the axes and the diagonals at infinity.
    out[0] = uint(as_type<ushort>(atan2(hy[0], hx[0])));   // atan2(+0, +1)
    out[1] = uint(as_type<ushort>(atan2(hy[1], hx[1])));   // atan2(-0, +1)
    out[2] = uint(as_type<ushort>(atan2(hy[2], hx[2])));   // atan2(+0, -1)
    out[3] = uint(as_type<ushort>(atan2(hy[3], hx[3])));   // atan2(-0, -1)
    out[4] = uint(as_type<ushort>(atan2(hy[4], hx[4])));   // atan2(+1, +0)
    out[5] = uint(as_type<ushort>(atan2(hy[5], hx[5])));   // atan2(-1, +0)
    out[6] = uint(as_type<ushort>(atan2(hy[6], hx[6])));   // atan2(+1, +inf)
    out[7] = uint(as_type<ushort>(atan2(hy[7], hx[7])));   // atan2(+1, -inf)
    out[8] = uint(as_type<ushort>(atan2(hy[8], hx[8])));   // atan2(+inf, +1)
    out[9] = uint(as_type<ushort>(atan2(hy[9], hx[9])));   // atan2(+inf, +inf)

    // air.atan2.f32 -- `precise::` because a fast-math compile spells the plain call
    // `air.fast_atan2.f32`, which is already covered.
    out[10] = as_type<uint>(precise::atan2(fy[0], fx[0]));   // atan2(+0, +1)
    out[11] = as_type<uint>(precise::atan2(fy[1], fx[1]));   // atan2(-0, +1)
    out[12] = as_type<uint>(precise::atan2(fy[2], fx[2]));   // atan2(+0, -1)
    out[13] = as_type<uint>(precise::atan2(fy[3], fx[3]));   // atan2(+1, +0)
    out[14] = as_type<uint>(precise::atan2(fy[4], fx[4]));   // atan2(+inf, +inf)
    out[15] = as_type<uint>(precise::atan2(fy[5], fx[5]));   // atan2(+1, +inf)
    out[16] = as_type<uint>(precise::atan2(fy[6], fx[6]));   // atan2(+1, -inf)
    out[17] = as_type<uint>(precise::atan2(fy[7], fx[7]));   // atan2(-0, -1)
}
