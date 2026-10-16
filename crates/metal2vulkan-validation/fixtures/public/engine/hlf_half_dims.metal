// Owned fixture: how Metal converts texels to `half` in the texture types
// `bcx_half_read` does not read. A 1D, a 1D array, a 2D array, a 3D, and a cube `half` texture
// (each case binds one format to all five, 16 texels wide) are each read at a texel and sampled
// through a nearest-filter and a linear-filter sampler, both clamp-to-edge with normalized
// coordinates; the 1D array is only sampled (the translation does not yet decode its read or
// width query, so its width is the constant 16). Lane `i` reads the texel-space coordinate
// `coord[i]` (`x`, `y`, `z`, and the layer or face in `w`) and the cube direction `dir[i]`, and
// writes fourteen results, widened to `float`, to `out[14i ..]` (per texture in slot order: read,
// nearest, linear; the 1D array nearest and linear). Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_edge(filter::nearest, address::clamp_to_edge);
constant sampler linear_edge(filter::linear, address::clamp_to_edge);

kernel void hlf_half_dims(texture1d<half, access::sample> t1 [[texture(0)]],
                          texture1d_array<half, access::sample> t1a [[texture(1)]],
                          texture2d_array<half, access::sample> t2a [[texture(2)]],
                          texture3d<half, access::sample> t3 [[texture(3)]],
                          texturecube<half, access::sample> tc [[texture(4)]],
                          const device float4 *coord [[buffer(0)]],
                          device float4 *out [[buffer(1)]],
                          const device float4 *dir [[buffer(2)]],
                          uint tid [[thread_position_in_grid]])
{
    float4 c = coord[tid];
    uint layer = uint(c.w);
    device float4 *o = out + 14 * tid;
    float x1 = c.x / float(t1.get_width());
    o[0] = float4(t1.read(uint(c.x)));
    o[1] = float4(t1.sample(nearest_edge, x1));
    o[2] = float4(t1.sample(linear_edge, x1));
    float x1a = c.x / 16.0;
    o[3] = float4(t1a.sample(nearest_edge, x1a, layer));
    o[4] = float4(t1a.sample(linear_edge, x1a, layer));
    float2 u2a = c.xy / float2(t2a.get_width(), t2a.get_height());
    o[5] = float4(t2a.read(uint2(c.xy), layer));
    o[6] = float4(t2a.sample(nearest_edge, u2a, layer));
    o[7] = float4(t2a.sample(linear_edge, u2a, layer));
    float3 u3 = c.xyz / float3(t3.get_width(), t3.get_height(), t3.get_depth());
    o[8] = float4(t3.read(uint3(c.xyz)));
    o[9] = float4(t3.sample(nearest_edge, u3));
    o[10] = float4(t3.sample(linear_edge, u3));
    float3 d = dir[tid].xyz;
    o[11] = float4(tc.read(uint2(c.xy), layer));
    o[12] = float4(tc.sample(nearest_edge, d));
    o[13] = float4(tc.sample(linear_edge, d));
}
