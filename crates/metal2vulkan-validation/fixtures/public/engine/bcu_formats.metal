// Owned fixture: Metal's bicubic filter (`filter::bicubic`) per bound pixel
// format. An impulse reading (`a2_bicubic_weights`) returned two linear weights on an
// `rgba32_float` texture, while a corpus case on `rgba16_float` returned Catmull-Rom weights, so
// the filter Metal applies may depend on the format bound. Ten textures are sampled at each lane's
// normalized coordinate through one clamp-to-edge bicubic sampler, in the form every corpus
// bicubic site uses (an implicit sample: in a kernel, the base level); the cases bind a different
// format to each slot. `t8` and `t9` are `half` textures (the corpus's `v4f16` samples), returned
// widened to `float`. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_edge(filter::bicubic, address::clamp_to_edge, coord::normalized);

kernel void bcu_formats(texture2d<float, access::sample> t0 [[texture(0)]],
                        texture2d<float, access::sample> t1 [[texture(1)]],
                        texture2d<float, access::sample> t2 [[texture(2)]],
                        texture2d<float, access::sample> t3 [[texture(3)]],
                        texture2d<float, access::sample> t4 [[texture(4)]],
                        texture2d<float, access::sample> t5 [[texture(5)]],
                        texture2d<float, access::sample> t6 [[texture(6)]],
                        texture2d<float, access::sample> t7 [[texture(7)]],
                        texture2d<half, access::sample> t8 [[texture(8)]],
                        texture2d<half, access::sample> t9 [[texture(9)]],
                        const device float2 *coord [[buffer(0)]],
                        device float4 *out [[buffer(1)]],
                        uint tid [[thread_position_in_grid]])
{
    float2 uv = coord[tid];
    out[10 * tid + 0] = t0.sample(bicubic_edge, uv);
    out[10 * tid + 1] = t1.sample(bicubic_edge, uv);
    out[10 * tid + 2] = t2.sample(bicubic_edge, uv);
    out[10 * tid + 3] = t3.sample(bicubic_edge, uv);
    out[10 * tid + 4] = t4.sample(bicubic_edge, uv);
    out[10 * tid + 5] = t5.sample(bicubic_edge, uv);
    out[10 * tid + 6] = t6.sample(bicubic_edge, uv);
    out[10 * tid + 7] = t7.sample(bicubic_edge, uv);
    out[10 * tid + 8] = float4(t8.sample(bicubic_edge, uv));
    out[10 * tid + 9] = float4(t9.sample(bicubic_edge, uv));
}
