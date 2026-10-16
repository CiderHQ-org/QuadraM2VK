// Owned fixture: the probe of the level of detail Metal derives from an
// explicit cube gradient, face by face. A sampler that minifies linearly and magnifies nearest
// over one level shows only whether the device minified, so sweeping a gradient's magnitude finds
// where the device's footprint crosses one texel. Lane i takes face i % 6, gradient axis
// (i / 6) % 6 (+x, -x, +y, -y, +z, -z), the gradient as dP/dx or dP/dy ((i / 36) % 2), and
// magnitude in[18 + i / 72].x; it samples a cube and layer 1 of a cube array at two directions
// on that face (in[face], in[6 + face]) and writes their red channels to out[i]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_cube_grad_lod(texturecube<float> tc [[texture(0)]],
                              texturecube_array<float> tca [[texture(1)]],
                              device const float4 *in [[buffer(0)]],
                              device float4 *out [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
    constexpr sampler color(coord::normalized, min_filter::linear, mag_filter::nearest,
                            mip_filter::none, address::clamp_to_edge);
    uint face = i % 6, axis = (i / 6) % 6, which = (i / 36) % 2, k = i / 72;
    float3 g = in[12 + axis].xyz * in[18 + k].x;
    float3 gx = which == 0 ? g : float3(0.0);
    float3 gy = which == 0 ? float3(0.0) : g;
    float3 d0 = in[face].xyz, d1 = in[6 + face].xyz;
    out[i] = float4(tc.sample(color, d0, gradientcube(gx, gy)).x,
                    tca.sample(color, d0, 1, gradientcube(gx, gy)).x,
                    tc.sample(color, d1, gradientcube(gx, gy)).x,
                    tca.sample(color, d1, 1, gradientcube(gx, gy)).x);
}
