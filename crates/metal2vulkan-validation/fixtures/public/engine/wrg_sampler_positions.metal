// Owned fixture: which texels Metal's samples and gathers read at
// the positions where its arithmetic decides them. Per axis Metal wraps a normalized coordinate
// into one period first, under repeat and mirrored repeat, then rounds its product with the size
// to `float`, ties away from zero; the filters read their footprints from that position. The cases
// bind odd sizes and read coordinates at texel centres and edges, a few ulps on either side, at the
// 1/512-texel steps of the filter's footprint, outside [0, 1), and at exact ties of the product.
// Texel (i, j) of the 2D texture `t` and of each layer `l` of the array `a` holds
// (i + 1, j + 1, l, 1) in a 32-bit float format; texel i of the 1D texture `s` holds (i + 1, 0, 0,
// 1). So a nearest sample and a gather of x or y name the column and row they read, a border texel
// reads 0, and a linear or bicubic sample weighs the indices of its footprint. Lane `i` reads the
// coordinate `coord[i]` and writes 20 results to `out[20i ..]` in the order below. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler linear_repeat(filter::linear, address::repeat);
constant sampler linear_mirror(filter::linear, address::mirrored_repeat);
constant sampler linear_zero(filter::linear, address::clamp_to_zero);
constant sampler nearest_repeat(filter::nearest, address::repeat);
constant sampler nearest_mirror(filter::nearest, address::mirrored_repeat);
constant sampler nearest_zero(filter::nearest, address::clamp_to_zero);
constant sampler nearest_edge(filter::nearest, address::clamp_to_edge);
constant sampler bicubic_repeat(filter::bicubic, address::repeat);
constant sampler bicubic_zero(filter::bicubic, address::clamp_to_zero);

kernel void wrg_sampler_positions(texture2d<float, access::sample> t [[texture(0)]],
                                  texture1d<float, access::sample> s [[texture(1)]],
                                  texture2d_array<float, access::sample> a [[texture(2)]],
                                  const device float2 *coord [[buffer(0)]],
                                  device float4 *out [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]])
{
    float2 c = coord[tid];
    device float4 *o = out + 20 * tid;
    o[0] = t.sample(linear_repeat, c);
    o[1] = t.sample(linear_mirror, c);
    o[2] = t.sample(nearest_repeat, c);
    o[3] = t.sample(nearest_mirror, c);
    o[4] = t.sample(linear_zero, c);
    o[5] = t.sample(nearest_edge, c);
    o[6] = t.sample(nearest_zero, c);
    o[7] = t.sample(bicubic_repeat, c);
    o[8] = t.sample(bicubic_zero, c);
    o[9] = t.gather(nearest_repeat, c, int2(0), component::x);
    o[10] = t.gather(nearest_repeat, c, int2(0), component::y);
    o[11] = t.gather(nearest_mirror, c, int2(0), component::x);
    o[12] = t.gather(nearest_mirror, c, int2(0), component::y);
    o[13] = t.gather(nearest_zero, c, int2(0), component::x);
    o[14] = t.gather(nearest_zero, c, int2(0), component::y);
    o[15] = t.gather(nearest_edge, c, int2(1, -1), component::x);
    o[16] = t.gather(nearest_edge, c, int2(1, -1), component::y);
    o[17] = s.sample(nearest_repeat, c.x);
    o[18] = s.sample(nearest_mirror, c.x);
    o[19] = a.gather(nearest_mirror, c, 1, int2(0), component::x);
}
