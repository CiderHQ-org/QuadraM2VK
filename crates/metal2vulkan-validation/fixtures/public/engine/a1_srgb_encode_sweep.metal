// Owned fixture: Metal's sRGB encode over a dense sweep. The translator encodes a color
// channel with the MSL transfer (`12.92 c` below 0.0031308, else `1.055 c^(1/2.4) - 0.055`) and
// rounds `c * 255` to nearest even; whether Metal rounds half up instead is open, and so is
// whether the encode, like the decode, goes through a table. Compiled with the default
// fast math, as the corpus's packs are.
//
// Thread t (0..5503) packs two vectors and writes two words:
// - `pack_float_to_srgb_unorm4x8` (`air.pack.unorm4x8.srgb.v4f32`) of red, green, and blue
//   `3t, 3t + 1, 3t + 2` times 2^-14 (every multiple of 2^-14 in [0, 1.0078], exact) and alpha
//   `3t * 2^-14` (a linear channel);
// - `pack_half_to_srgb_unorm4x8` (`air.pack.unorm4x8.srgb.v4f16`) of the half patterns
//   `3t, 3t + 1, 3t + 2` (every half in [0, 1], and on to 2.25) and alpha the pattern `3t`.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a1_srgb_encode_sweep(
    device uint *out [[buffer(0)]],
    uint t [[thread_position_in_grid]])
{
    uint k = 3u * t;
    const float step = 0x1p-14f;
    float4 c = float4(float(k) * step, float(k + 1u) * step, float(k + 2u) * step, float(k) * step);
    half4 h = half4(as_type<half>(ushort(k)), as_type<half>(ushort(k + 1u)),
                    as_type<half>(ushort(k + 2u)), as_type<half>(ushort(k)));
    out[2 * t + 0] = pack_float_to_srgb_unorm4x8(c);
    out[2 * t + 1] = pack_half_to_srgb_unorm4x8(h);
}
