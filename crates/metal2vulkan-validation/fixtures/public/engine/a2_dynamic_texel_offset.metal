// Owned fixture: dynamic texel offsets on a sample and a gather.
// MSL 6.13 requires an offset in [-8, 7]; the lowering clamps a dynamic offset into that
// range (`SClamp(offset, -8, 7)`), which is the identity on every offset Metal defines. The
// offsets come from the input buffer, so one case reads in-range texel values and another reads
// what Apple GPUs return outside the range. A nearest sampler at a texel centre of a texture
// whose texel values are their own coordinates makes the effective offset readable. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_clamp(filter::nearest, address::clamp_to_edge, coord::normalized);

kernel void a2_dynamic_texel_offset(texture2d<float, access::sample> t [[texture(0)]],
                                    const device int2 *offsets [[buffer(0)]],
                                    device float4 *out [[buffer(1)]],
                                    uint tid [[thread_position_in_grid]])
{
    int2 o = offsets[tid];
    float2 uv = float2(32.5f / 64.0f, 32.5f / 64.0f);
    out[2 * tid + 0] = t.sample(nearest_clamp, uv, level(0), o);
    out[2 * tid + 1] = t.gather(nearest_clamp, uv, o);
}
