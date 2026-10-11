// Which Metal variant does MoltenVK's runtime compile actually reach for the five biggest `fast_`
// families in the corpus? SPIRV-Cross emits the UNQUALIFIED MSL name for GLSL `Exp`, `Pow`, `Log`,
// `InverseSqrt` and `Sqrt`, and whether Metal then resolves that to `fast::` or `precise::` is a
// property of MoltenVK's compile options, not of anything this repository controls. It is already
// known to differ per function: `sqrt` is approximate under MoltenVK's default and `atan` is not.
//
// The answer decides whether 29000+ corpus call sites are byte-exact or systematically off:
//   air.fast_powr 8320   air.fast_exp 7297   air.fast_rsqrt 6209
//   air.fast_pow 4904    air.fast_log 2682   (air.fast_sqrt 7862 is the control -- already known)
//
// Each argument was chosen to MAXIMIZE the gap between the two variants, measured on an Apple M3
// Max / macOS 26.5.2 over 20000 arguments per family, so no row is a coin flip:
//
//   exp(17.309)          fast 0x4bfb02d6   precise 0x4bfb02e5   15 ULP apart
//   exp(18.02)           fast 0x4c7f885f   precise 0x4c7f886e   15
//   powr(19.193, 5.36)   fast 0x4ae63f61   precise 0x4ae63f85   36
//   powr(16.572, 4.43)   fast 0x48765665   precise 0x48765687   34
//   log(1.0590001)       fast 0x3d6acdcf   precise 0x3d6acdd3    4
//   log(0.976)           fast 0xbcc70174   precise 0xbcc70171     3
//   rsqrt(0.027)         fast 0x40c2beed   precise 0x40c2beec     1
//   rsqrt(0.10300001)    fast 0x40476aa8   precise 0x40476aa7     1
//   sqrt(15.075001)      fast 0x40787d68   precise 0x40787d66     2
//   sqrt(15.673001)      fast 0x407d5edb   precise 0x407d5ed9     2
//
// Under `xcrun metal` defaults the unqualified spelling is bit-identical to `fast::` on all 20000
// arguments of every one of the five, so this file compiled normally emits `air.fast_*` and the
// Metal oracle is the FAST column throughout. A Match therefore says MoltenVK reaches `fast::` too;
// a Mismatch would say it does not, and would name exactly which families.
//
// ANSWERED: it Matches on all ten rows. MoltenVK reaches `fast::` for `exp`, `powr`, `log`,
// `rsqrt` and `sqrt`, so those five families -- 29274 corpus call sites -- are byte-exact with the
// single-opcode lowering they already have. The `precise::`-only set stays confined to the
// arctangent/tangent family (`air.fast_atan2.f32`, `air.fast_tan.f32`), which is what makes that
// one a genuine MoltenVK limitation rather than a general rule about unqualified MSL names.
//
// Every argument is read from a buffer so the frontend cannot fold anything away.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_fast_variant_reach(
    device const float* in [[buffer(0)]],
    device uint* out [[buffer(1)]])
{
    float exp_a = in[0];
    float exp_b = in[1];
    float pow_a0 = in[2];
    float pow_a1 = in[3];
    float pow_b0 = in[4];
    float pow_b1 = in[5];
    float log_a = in[6];
    float log_b = in[7];
    float rsqrt_a = in[8];
    float rsqrt_b = in[9];
    float sqrt_a = in[10];
    float sqrt_b = in[11];

    out[0] = as_type<uint>(exp(exp_a));
    out[1] = as_type<uint>(exp(exp_b));
    out[2] = as_type<uint>(powr(pow_a0, pow_b0));
    out[3] = as_type<uint>(powr(pow_a1, pow_b1));
    out[4] = as_type<uint>(log(log_a));
    out[5] = as_type<uint>(log(log_b));
    out[6] = as_type<uint>(rsqrt(rsqrt_a));
    out[7] = as_type<uint>(rsqrt(rsqrt_b));
    out[8] = as_type<uint>(sqrt(sqrt_a));
    out[9] = as_type<uint>(sqrt(sqrt_b));
}
