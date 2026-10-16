// Owned fixture: which cube face and texel Metal reads
// when two or three of a direction's components share the largest magnitude, or lie at either end
// of the float range. The cases bind an rgba32_float cube whose texel (i, j) of face f holds
// (f + 1, i + 1, j + 1, 0), so a result names the face and the texel it read. Lane `i` samples the direction `dir[i].xyz` through a nearest sampler, and
// gathers the red, green and blue components at the direction `gdir[i].xyz`, which ties two axes
// only where the 2x2 block stays off the cube's corners (a corner texel is the device's choice in
// Vulkan); it writes 4 results to `out[4i ..]` in that order. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_edge(filter::nearest, address::clamp_to_edge);

kernel void wrg_cube_ties(texturecube<float, access::sample> t [[texture(0)]],
                          const device float4 *dir [[buffer(0)]],
                          const device float4 *gdir [[buffer(2)]],
                          device float4 *out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]])
{
    float3 d = dir[tid].xyz;
    float3 g = gdir[tid].xyz;
    device float4 *o = out + 4 * tid;
    o[0] = t.sample(nearest_edge, d);
    o[1] = t.gather(nearest_edge, g, component::x);
    o[2] = t.gather(nearest_edge, g, component::y);
    o[3] = t.gather(nearest_edge, g, component::z);
}
