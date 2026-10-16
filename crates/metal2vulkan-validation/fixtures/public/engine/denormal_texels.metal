// Owned fixture: plain gathers and nearest samples of
// 32-bit float textures whose texels include denormals of both signs. Lane i reads at in[i].xy
// through a nearest clamp-to-edge sampler and a nearest repeating sampler, without a mip filter,
// and writes eight float4s from out[8i]: gathers of an r32Float texture (texture 0) and of an
// rgba32Float texture's y and w channels (texture 1), nearest samples of both (one with a texel
// offset), and a nearest sample of layer 1 of an r32Float 2D array (texture 2). Written for this
// repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void denormal_texels(texture2d<float> r [[texture(0)]],
                            texture2d<float> rgba [[texture(1)]],
                            texture2d_array<float> arr [[texture(2)]],
                            device const float4 *in [[buffer(0)]],
                            device float4 *out [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    constexpr sampler edge(coord::normalized, filter::nearest, address::clamp_to_edge);
    constexpr sampler wrap(coord::normalized, filter::nearest, address::repeat);
    float2 uv = in[i].xy;
    out[8 * i + 0] = r.gather(edge, uv);
    out[8 * i + 1] = r.sample(edge, uv);
    out[8 * i + 2] = rgba.gather(edge, uv, int2(0), component::y);
    out[8 * i + 3] = rgba.gather(wrap, uv, int2(0), component::w);
    out[8 * i + 4] = rgba.sample(edge, uv);
    out[8 * i + 5] = r.gather(wrap, uv);
    out[8 * i + 6] = rgba.sample(wrap, uv, int2(1, -1));
    out[8 * i + 7] = arr.sample(edge, uv, 1);
}
