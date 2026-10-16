// Owned fixture: the LOD queries the decoder once lacked -- the
// clamped and unclamped LOD of 2D array, 3D, cube, and cube array textures and of depth 2D,
// 2D array, cube, and cube array textures -- in a fragment function, at coordinates linear in
// the fragment's position: the plane coordinate pos.xy / 16 (a 2D array and a 3D texture, whose
// third coordinate is constant) and the direction (1, (8 - pos.y) / 8, (8 - pos.x) / 8), on the
// +X face of a cube. The fragment at (x, y) of a 16-wide target writes the sixteen values to
// out[4(16y + x) .. 4(16y + x) + 3] and returns zero. Loop-free.
#include <metal_stdlib>
using namespace metal;

fragment float4 unk_calculate_lod(float4 pos [[position]],
                                  texture2d_array<float> t2a [[texture(0)]],
                                  texture3d<float> t3 [[texture(1)]],
                                  texturecube<float> tc [[texture(2)]],
                                  texturecube_array<float> tca [[texture(3)]],
                                  depth2d<float> d2 [[texture(4)]],
                                  depth2d_array<float> d2a [[texture(5)]],
                                  depthcube<float> dc [[texture(6)]],
                                  depthcube_array<float> dca [[texture(7)]],
                                  sampler s [[sampler(0)]],
                                  device float4 *out [[buffer(0)]]) {
    float2 uv = pos.xy / 16.0;
    float3 uvw = float3(uv, 0.5);
    float3 d = float3(1.0, (8.0 - pos.y) / 8.0, (8.0 - pos.x) / 8.0);
    uint at = 4 * (16 * uint(pos.y) + uint(pos.x));
    out[at] = float4(t2a.calculate_clamped_lod(s, uv), t2a.calculate_unclamped_lod(s, uv),
                     t3.calculate_clamped_lod(s, uvw), t3.calculate_unclamped_lod(s, uvw));
    out[at + 1] = float4(tc.calculate_clamped_lod(s, d), tc.calculate_unclamped_lod(s, d),
                         tca.calculate_clamped_lod(s, d), tca.calculate_unclamped_lod(s, d));
    out[at + 2] = float4(d2.calculate_clamped_lod(s, uv), d2.calculate_unclamped_lod(s, uv),
                         d2a.calculate_clamped_lod(s, uv), d2a.calculate_unclamped_lod(s, uv));
    out[at + 3] = float4(dc.calculate_clamped_lod(s, d), dc.calculate_unclamped_lod(s, d),
                         dca.calculate_clamped_lod(s, d), dca.calculate_unclamped_lod(s, d));
    return float4(0.0);
}
