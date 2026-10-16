// Owned fixture: Metal's `half` math functions without fast math
// (`-fno-fast-math`), at the lanes where a device or the translation's `float` routines rounded to
// `half` were seen to answer otherwise: the zeros' signs of `sin`, `tan`, `asin`, and `atan`;
// `fmod(x, ±inf)`; `fmod(1.5, -0.75)`; `sqrt(65504)`; and `atan2`, `pow`, and `powr` at (3, 7) and
// (-0.300048828125, 0.7001953125). `trn_half_fast` is the same program in the other mode.
// metal-fast-math: off
//
// Thread i reads `in[i]`, the bits of two halves (a in the low 16 bits, b in the high), and
// writes 8 words, two halves each (the first in the low 16 bits), `out[8i .. 8i + 7]`:
// (sin a, cos a), (tan a, asin a), (acos a, atan a), (atan2(a, b), fmod(a, b)), (sqrt a, rsqrt a),
// (pow(a, b), powr(a, b)), (sinpi a, cospi a), and (tanpi a, b). Loop-free.
#include <metal_stdlib>
using namespace metal;

static uint p(half lo, half hi)
{
    return uint(as_type<ushort>(lo)) | (uint(as_type<ushort>(hi)) << 16);
}

kernel void trn_half_precise(device const uint *in [[buffer(0)]],
                 device uint *out [[buffer(1)]],
                 uint i [[thread_position_in_grid]])
{
    uint w = in[i];
    half a = as_type<half>(ushort(w & 0xFFFF));
    half b = as_type<half>(ushort(w >> 16));
    out[8 * i + 0] = p(sin(a), cos(a));
    out[8 * i + 1] = p(tan(a), asin(a));
    out[8 * i + 2] = p(acos(a), atan(a));
    out[8 * i + 3] = p(atan2(a, b), fmod(a, b));
    out[8 * i + 4] = p(sqrt(a), rsqrt(a));
    out[8 * i + 5] = p(pow(a, b), powr(a, b));
    out[8 * i + 6] = p(sinpi(a), cospi(a));
    out[8 * i + 7] = p(tanpi(a), b);
}
