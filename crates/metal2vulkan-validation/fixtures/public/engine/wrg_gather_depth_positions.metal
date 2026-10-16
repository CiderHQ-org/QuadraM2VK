// Owned fixture: which 2x2 block Metal's depth gathers and
// `gather_compare` read at the positions where its arithmetic decides them (see
// `wrg_sampler_positions`: the wrap into one period, the product rounded ties away from zero, and
// the footprint's 1/512-texel steps). The cases bind a 32-bit float depth texture of odd size whose
// texel (i, j) holds (i + 1 + 256 (j + 1)) / 2^20, so a gather names the texels it reads and a
// comparison against `ref[i]` tells them apart from their neighbours. Lane `i` reads the coordinate
// `coord[i]` and writes 8 results to `out[8i ..]` in the order below. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler depth_edge(filter::nearest, address::clamp_to_edge);
constant sampler depth_zero(filter::nearest, address::clamp_to_zero);
constant sampler depth_repeat(filter::nearest, address::repeat);
constant sampler depth_mirror(filter::nearest, address::mirrored_repeat);
constant sampler less_edge(filter::nearest, address::clamp_to_edge, compare_func::less);
constant sampler less_zero(filter::nearest, address::clamp_to_zero, compare_func::less);
constant sampler less_repeat(filter::nearest, address::repeat, compare_func::less);

kernel void wrg_gather_depth_positions(depth2d<float, access::sample> t [[texture(0)]],
                                       const device float2 *coord [[buffer(0)]],
                                       const device float *ref [[buffer(2)]],
                                       device float4 *out [[buffer(1)]],
                                       uint tid [[thread_position_in_grid]])
{
    float2 c = coord[tid];
    float r = ref[tid];
    device float4 *o = out + 8 * tid;
    o[0] = t.gather(depth_edge, c);
    o[1] = t.gather(depth_zero, c);
    o[2] = t.gather(depth_repeat, c);
    o[3] = t.gather(depth_mirror, c);
    o[4] = t.gather_compare(less_edge, c, r);
    o[5] = t.gather_compare(less_zero, c, r);
    o[6] = t.gather_compare(less_repeat, c, r, int2(1, -1));
    o[7] = t.gather(depth_edge, c, int2(-1, 1));
}
