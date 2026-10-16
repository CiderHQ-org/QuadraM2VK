// Owned fixture: the second probe of the level of detail Metal derives
// from an explicit cube gradient, driven by a table. Buffer 0 holds the configuration count C in
// the bits of in[0].x, then for each configuration c a direction in[1 + 3c] and the two
// gradients in[2 + 3c] (dP/dx) and in[3 + 3c] (dP/dy), then the magnitudes from in[1 + 3C]. Lane
// i takes configuration i % C and scales both gradients by magnitude i / C; it samples a cube and
// layer 1 of a cube array through a sampler that minifies linearly and magnifies nearest over
// one level, and writes the two red channels to out[i]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_cube_grad_table(texturecube<float> tc [[texture(0)]],
                                texturecube_array<float> tca [[texture(1)]],
                                device const float4 *in [[buffer(0)]],
                                device float2 *out [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    constexpr sampler color(coord::normalized, min_filter::linear, mag_filter::nearest,
                            mip_filter::none, address::clamp_to_edge);
    uint n = as_type<uint>(in[0].x);
    uint c = i % n, k = i / n;
    float m = in[1 + 3 * n + k].x;
    float3 d = in[1 + 3 * c].xyz;
    float3 gx = in[2 + 3 * c].xyz * m, gy = in[3 + 3 * c].xyz * m;
    out[i] = float2(tc.sample(color, d, gradientcube(gx, gy)).x,
                    tca.sample(color, d, 1, gradientcube(gx, gy)).x);
}
