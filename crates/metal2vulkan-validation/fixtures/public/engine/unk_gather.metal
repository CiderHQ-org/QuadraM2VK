// Owned fixture: the gathers the decoder once lacked -- of a cube
// and a cube array texture (components x, w, and z), of a depth cube and cube array, and the
// comparison gathers of depth 2D, 2D array, cube, and cube array textures. Each lane reads its
// 2D coordinate and comparison reference, layer, and cube direction from in[i] and writes nine
// float4s to out[9i .. 9i + 8]. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Lane {
    float4 p;   // 2D coordinate (xy), comparison reference (z), array layer (w)
    float4 dir; // cube direction (xyz)
};

kernel void unk_gather(texturecube<float> tc [[texture(0)]],
                       texturecube_array<float> tca [[texture(1)]],
                       depth2d<float> d2 [[texture(2)]],
                       depth2d_array<float> d2a [[texture(3)]],
                       depthcube<float> dc [[texture(4)]],
                       depthcube_array<float> dca [[texture(5)]],
                       device const Lane *in [[buffer(0)]],
                       device float4 *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    constexpr sampler near(coord::normalized, filter::nearest, mip_filter::none,
                           address::clamp_to_edge);
    constexpr sampler cmp(coord::normalized, filter::nearest, mip_filter::none,
                          address::clamp_to_edge, compare_func::less_equal);
    Lane l = in[i];
    float2 p = l.p.xy;
    float ref = l.p.z;
    uint layer = uint(l.p.w);
    float3 d = l.dir.xyz;
    out[9 * i] = tc.gather(near, d, component::x);
    out[9 * i + 1] = tc.gather(near, d, component::w);
    out[9 * i + 2] = tca.gather(near, d, layer, component::z);
    out[9 * i + 3] = dc.gather(near, d);
    out[9 * i + 4] = dca.gather(near, d, layer);
    out[9 * i + 5] = d2.gather_compare(cmp, p, ref);
    out[9 * i + 6] = d2a.gather_compare(cmp, p, layer, ref);
    out[9 * i + 7] = dc.gather_compare(cmp, d, ref);
    out[9 * i + 8] = dca.gather_compare(cmp, d, layer, ref);
}
