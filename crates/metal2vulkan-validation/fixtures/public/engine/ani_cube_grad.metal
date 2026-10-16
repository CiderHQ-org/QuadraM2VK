// Owned fixture: explicit cube gradients through an anisotropic
// constexpr sampler. Lane i samples a cube (texture 0) and layer 1 of a cube array (texture 1) at
// direction in[3i] with gradients in[3i + 1] (dP/dx) and in[3i + 2] (dP/dy), through a sampler that
// filters linearly with a linear mip filter, clamps to the edge, and takes up to 16 anisotropic
// probes, and writes the two samples to out[2i] and out[2i + 1]. Written for this repository; no
// third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ani_cube_grad(texturecube<float> tc [[texture(0)]],
                          texturecube_array<float> tca [[texture(1)]],
                          device const float4 *in [[buffer(0)]],
                          device float4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    constexpr sampler aniso(coord::normalized, filter::linear, mip_filter::linear,
                            address::clamp_to_edge, max_anisotropy(16));
    float3 d = in[3 * i].xyz, gx = in[3 * i + 1].xyz, gy = in[3 * i + 2].xyz;
    out[2 * i] = tc.sample(aniso, d, gradientcube(gx, gy));
    out[2 * i + 1] = tca.sample(aniso, d, 1, gradientcube(gx, gy));
}
