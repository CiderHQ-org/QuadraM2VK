// Owned fixture: the samples the decoder once lacked -- gradient
// samples of 3D, cube, and cube array textures, of depth 2D, 2D array, cube, and cube array
// textures with and without comparison, and the depth cube array sample and depth cube and cube
// array comparisons at level 0. Each lane reads its coordinates, gradients, layer, and comparison
// reference from in[i] and writes six float4s to out[6i .. 6i + 5]. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Lane {
    float4 p;   // 3D and 2D coordinate (xyz), comparison reference (w)
    float4 dx;  // 3D and 2D dP/dx (xyz), array layer (w)
    float4 dy;  // 3D and 2D dP/dy (xyz)
    float4 dir; // cube direction (xyz)
    float4 dcx; // cube dP/dx (xyz)
    float4 dcy; // cube dP/dy (xyz)
};

kernel void unk_sample_grad(texture3d<float> t3 [[texture(0)]],
                            texturecube<float> tc [[texture(1)]],
                            texturecube_array<float> tca [[texture(2)]],
                            depth2d<float> d2 [[texture(3)]],
                            depth2d_array<float> d2a [[texture(4)]],
                            depthcube<float> dc [[texture(5)]],
                            depthcube_array<float> dca [[texture(6)]],
                            device const Lane *in [[buffer(0)]],
                            device float4 *out [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    // Minification is linear and magnification nearest, so the gradients choose the filter.
    constexpr sampler color(coord::normalized, min_filter::linear, mag_filter::nearest,
                            mip_filter::none, address::clamp_to_edge);
    constexpr sampler near(coord::normalized, filter::nearest, mip_filter::none,
                           address::clamp_to_edge);
    constexpr sampler cmp(coord::normalized, filter::nearest, mip_filter::none,
                          address::clamp_to_edge, compare_func::less_equal);
    Lane l = in[i];
    float3 p = l.p.xyz, dx = l.dx.xyz, dy = l.dy.xyz;
    float3 d = l.dir.xyz, cx = l.dcx.xyz, cy = l.dcy.xyz;
    float ref = l.p.w;
    uint layer = uint(l.dx.w);
    out[6 * i] = t3.sample(color, p, gradient3d(dx, dy));
    out[6 * i + 1] = tc.sample(color, d, gradientcube(cx, cy));
    out[6 * i + 2] = tca.sample(color, d, layer, gradientcube(cx, cy));
    out[6 * i + 3] = float4(d2.sample(near, p.xy, gradient2d(dx.xy, dy.xy)),
                            d2a.sample(near, p.xy, layer, gradient2d(dx.xy, dy.xy)),
                            dc.sample(near, d, gradientcube(cx, cy)),
                            dca.sample(near, d, layer, gradientcube(cx, cy)));
    out[6 * i + 4] = float4(d2.sample_compare(cmp, p.xy, ref, gradient2d(dx.xy, dy.xy)),
                            d2a.sample_compare(cmp, p.xy, layer, ref, gradient2d(dx.xy, dy.xy)),
                            dc.sample_compare(cmp, d, ref, gradientcube(cx, cy)),
                            dca.sample_compare(cmp, d, layer, ref, gradientcube(cx, cy)));
    out[6 * i + 5] = float4(dca.sample(near, d, layer, level(0)),
                            dc.sample_compare(cmp, d, ref, level(0)),
                            dca.sample_compare(cmp, d, layer, ref, level(0)),
                            dca.sample(near, d, layer));
}
