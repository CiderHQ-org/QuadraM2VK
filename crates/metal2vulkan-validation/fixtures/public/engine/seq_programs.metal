// Owned fixture: the functions the seeded differential sequences of the
// parity harness run (`validation/src/parity/sequences/`), written for them. Integer-exact and
// loop-free: a kernel copies or mixes one word, and a draw covers whole pixels of an 8x8 target
// with colours a buffer holds, and a kernel reads or writes texels of an 8-bit unorm texture
// through 8-bit unorm words (generator round 2), so every byte a sequence ends with is
// defined, and a CPU model of the sequence computes it.
#include <metal_stdlib>
using namespace metal;

// `to[0] = from[0]`.
kernel void seq_copy(const device uint *from [[buffer(0)]], device uint *to [[buffer(1)]]) {
    to[0] = from[0];
}

// `word[0]` mixed with `salt[0]`, in place: the word it ends with depends on the order of the
// writes before.
kernel void seq_mix(device uint *word [[buffer(0)]], const device uint *salt [[buffer(1)]]) {
    word[0] = (word[0] ^ salt[0]) * 0x9E3779B1u + 0x7F4A7C15u;
}

struct SeqFragmentIn {
    float4 position [[position]];
    uint instance [[flat]];
};

// Instance `instance`'s rectangle `rects[instance]`, `(x0, y0, x1, y1)` in whole pixels of an
// 8x8 target (y down), as two triangles over its corners: vertices 0 to 2 are corners 0 to 2,
// vertices 3 to 5 are corners 1 to 3 (corner bit 0 picks x1, bit 1 picks y1). Every edge lies on
// a pixel boundary, so each pixel's centre is inside or outside on every rasterizer.
vertex SeqFragmentIn seq_rect(uint vertex_id [[vertex_id]], uint instance [[instance_id]],
                              const device uint4 *rects [[buffer(0)]]) {
    uint corner = vertex_id < 3u ? vertex_id : vertex_id - 2u;
    uint4 rect = rects[instance];
    float x = float((corner & 1u) != 0u ? rect.z : rect.x);
    float y = float((corner & 2u) != 0u ? rect.w : rect.y);
    SeqFragmentIn out;
    out.position = float4(x / 4.0 - 1.0, 1.0 - y / 4.0, 0.0, 1.0);
    out.instance = instance;
    return out;
}

// The colour `colors[instance]`: its four bytes are the texel's (an 8-bit unorm target, red in
// the low byte).
fragment float4 seq_color(SeqFragmentIn in [[stage_in]], const device uint *colors [[buffer(0)]]) {
    return unpack_unorm4x8_to_float(colors[in.instance]);
}

// The texel each thread of the grid names, read through `texels` (its format and swizzle) and
// packed into the word of `words` its place names, eight words a row: red in the low byte.
kernel void seq_texel_read(texture2d<float, access::read> texels [[texture(0)]],
                           device uint *words [[buffer(0)]],
                           uint2 at [[thread_position_in_grid]]) {
    words[at.y * 8u + at.x] = pack_float_to_unorm4x8(texels.read(at));
}

// The word of `words` each thread of the grid names, eight words a row, written as the texel its
// place names through `texels` (its format): red from the low byte.
kernel void seq_texel_write(texture2d<float, access::write> texels [[texture(0)]],
                            const device uint *words [[buffer(0)]],
                            uint2 at [[thread_position_in_grid]]) {
    texels.write(unpack_unorm4x8_to_float(words[at.y * 8u + at.x]), at);
}
