// Owned fixture: Metal's linear filter with `float` and `half` results in
// 1D and 2D, at the texture's edges included. A 2D and a 1D `float` texture and a 2D and a 1D
// `half` texture (each case binds one format to all four, 16 texels wide) are each sampled
// through a linear clamp-to-edge and a linear clamp-to-zero sampler with normalized
// coordinates: lane `i` reads the texel-space point `coord[i]` (`x`, `y`), divides it by the
// size, and samples the 1D textures at `x` alone. Lane `i` writes its eight results (per texture
// in slot order: edge, zero), the `half` ones widened to `float`, to `out[8i .. 8i + 7]`.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler linear_edge(filter::linear, address::clamp_to_edge);
constant sampler linear_zero(filter::linear, address::clamp_to_zero);

kernel void hlf_float_linear(texture2d<float, access::sample> f2 [[texture(0)]],
                             texture1d<float, access::sample> f1 [[texture(1)]],
                             texture2d<half, access::sample> h2 [[texture(2)]],
                             texture1d<half, access::sample> h1 [[texture(3)]],
                             const device float2 *coord [[buffer(0)]],
                             device float4 *out [[buffer(1)]],
                             uint tid [[thread_position_in_grid]])
{
    float2 p = coord[tid];
    float2 u2 = p / float2(f2.get_width(), f2.get_height());
    float u1 = p.x / float(f1.get_width());
    device float4 *o = out + 8 * tid;
    o[0] = f2.sample(linear_edge, u2);
    o[1] = f2.sample(linear_zero, u2);
    o[2] = f1.sample(linear_edge, u1);
    o[3] = f1.sample(linear_zero, u1);
    float2 v2 = p / float2(h2.get_width(), h2.get_height());
    float v1 = p.x / float(h1.get_width());
    o[4] = float4(h2.sample(linear_edge, v2));
    o[5] = float4(h2.sample(linear_zero, v2));
    o[6] = float4(h1.sample(linear_edge, v1));
    o[7] = float4(h1.sample(linear_zero, v1));
}
