// Owned fixture: which sample forms and texture types Metal's bicubic
// filter applies to. The front end admits a bicubic sampler only with both filters bicubic and no
// mip filter, so the level of detail selects neither a level nor a filter; this checks that. Each
// lane samples a one-level 2D texture implicitly (the corpus form), at `level(0)`, at
// `level(2.5)`, with `bias(2.5)` and `bias(-2.5)`, and with large gradients; a 2D array's layer 1
// implicitly and at `level(0)`; a 1D texture; and the 2D texture with a constant texel offset of
// (1, -1). Not `min_lod_clamp`: Apple's device compiler (M3 Max, macOS 26.7) fails to build any
// pipeline that samples through a bicubic sampler with it (XPC_ERROR_CONNECTION_INTERRUPTED), so
// Metal gives that form no result to read. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_edge(filter::bicubic, address::clamp_to_edge, coord::normalized);

kernel void bcu_forms(texture2d<float, access::sample> t [[texture(0)]],
                      texture2d_array<float, access::sample> a [[texture(1)]],
                      texture1d<float, access::sample> l [[texture(2)]],
                      const device float2 *coord [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      uint tid [[thread_position_in_grid]])
{
    float2 uv = coord[tid];
    out[10 * tid + 0] = t.sample(bicubic_edge, uv);
    out[10 * tid + 1] = t.sample(bicubic_edge, uv, level(0.0f));
    out[10 * tid + 2] = t.sample(bicubic_edge, uv, level(2.5f));
    out[10 * tid + 3] = t.sample(bicubic_edge, uv, bias(2.5f));
    out[10 * tid + 4] = t.sample(bicubic_edge, uv, gradient2d(float2(0.5f, 0.0f), float2(0.0f, 0.5f)));
    out[10 * tid + 5] = t.sample(bicubic_edge, uv, bias(-2.5f));
    out[10 * tid + 6] = a.sample(bicubic_edge, uv, 1);
    out[10 * tid + 7] = a.sample(bicubic_edge, uv, 1, level(0.0f));
    out[10 * tid + 8] = l.sample(bicubic_edge, uv.x);
    out[10 * tid + 9] = t.sample(bicubic_edge, uv, int2(1, -1));
}
