// Owned fixture: Metal's sRGB decode of every byte. The translator lowers
// `air.unpack.unorm4x8.srgb` as the table step `rint(decode(b / 255) * 4095) / 4095`, correctly
// rounded, from the old engine's device measurement and three observed steps (1, 1024, 4095 in
// case srgb-transfer-and-its-twelve-bit-table). A byte reaches 256 of the 4096 steps, so decoding
// every byte observes every step the conversion can produce. Compiled with the default fast math,
// as the corpus's unpacks are.
//
// Thread t (0..255) unpacks the word whose red, green, blue, and alpha bytes are t, 255 - t,
// t ^ 0x5a, and (7t + 3) & 255, so each color channel meets every byte once and alpha (linear)
// does too. It writes six words: the four `float` results of `unpack_unorm4x8_srgb_to_float`
// (`air.unpack.unorm4x8.srgb.v4f32`), then the four `half` results of
// `unpack_unorm4x8_srgb_to_half` (`air.unpack.unorm4x8.srgb.v4f16`) as two words, lane x in the
// low half of the first. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a1_srgb_decode_all_bytes(
    device uint *out [[buffer(0)]],
    uint t [[thread_position_in_grid]])
{
    uint w = (t & 255u) | ((255u - (t & 255u)) << 8) | (((t & 255u) ^ 0x5au) << 16)
        | (((7u * t + 3u) & 255u) << 24);
    float4 f = unpack_unorm4x8_srgb_to_float(w);
    half4 h = unpack_unorm4x8_srgb_to_half(w);
    device uint *o = out + 6 * t;
    o[0] = as_type<uint>(f.x);
    o[1] = as_type<uint>(f.y);
    o[2] = as_type<uint>(f.z);
    o[3] = as_type<uint>(f.w);
    o[4] = as_type<uint2>(h).x;
    o[5] = as_type<uint2>(h).y;
}
