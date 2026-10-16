// Owned fixture: `ani_cube_grad`'s samples in a
// fragment function, so a graphics pipeline hands the explicit cube gradients to the GPU. Pixel
// (i, 0) samples a cube (texture 0) and pixel (i, 1) layer 1 of a cube array (texture 1) at
// direction in[3i] with gradients in[3i + 1] (dP/dx) and in[3i + 2] (dP/dy), through
// `ani_cube_grad`'s sampler: linear filters, a linear mip filter, clamp to the edge, and up to 16
// anisotropic probes. Drawn by the fullscreen triangle `rtl_cube_cover` into an `rgba32Float`
// target, so pixel (i, 0) is the kernel's out[2i] and pixel (i, 1) its out[2i + 1]. Written for
// this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct RtlCubePoint {
    float4 position [[position]];
};

// The fullscreen triangle, (-1, -1), (3, -1), (-1, 3), computed rather than indexed from a local
// array (whose lifetime markers LLVM majors print differently).
vertex RtlCubePoint rtl_cube_cover(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return RtlCubePoint { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

fragment float4 rtl_cube_grad_fragment(RtlCubePoint p [[stage_in]],
                                       texturecube<float> tc [[texture(0)]],
                                       texturecube_array<float> tca [[texture(1)]],
                                       device const float4 *in [[buffer(0)]]) {
    constexpr sampler aniso(coord::normalized, filter::linear, mip_filter::linear,
                            address::clamp_to_edge, max_anisotropy(16));
    uint i = uint(p.position.x);
    float3 d = in[3 * i].xyz, gx = in[3 * i + 1].xyz, gy = in[3 * i + 2].xyz;
    if (uint(p.position.y) == 0u) {
        return tc.sample(aniso, d, gradientcube(gx, gy));
    }
    return tca.sample(aniso, d, 1, gradientcube(gx, gy));
}
