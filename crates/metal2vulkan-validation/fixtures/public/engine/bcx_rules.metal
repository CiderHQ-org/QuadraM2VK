// Owned fixture: how Metal's bicubic filter (`filter::bicubic`) sums and
// rounds per bound format. One clamp-to-edge bicubic sampler with pixel coordinates samples
// sixteen textures at each lane's coordinate: eight 2D `float` textures, four 2D `half`
// textures (returned widened to `float`), three 1D `float` textures and one 1D `half` texture
// (at the coordinate's `x`). The cases bind a different format and contents to each slot. Lane
// `i` writes its sixteen samples to `out[16i .. 16i + 15]` in slot order. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_pixel(filter::bicubic, address::clamp_to_edge, coord::pixel);

kernel void bcx_rules(texture2d<float, access::sample> f0 [[texture(0)]],
                      texture2d<float, access::sample> f1 [[texture(1)]],
                      texture2d<float, access::sample> f2 [[texture(2)]],
                      texture2d<float, access::sample> f3 [[texture(3)]],
                      texture2d<float, access::sample> f4 [[texture(4)]],
                      texture2d<float, access::sample> f5 [[texture(5)]],
                      texture2d<float, access::sample> f6 [[texture(6)]],
                      texture2d<float, access::sample> f7 [[texture(7)]],
                      texture2d<half, access::sample> h0 [[texture(8)]],
                      texture2d<half, access::sample> h1 [[texture(9)]],
                      texture2d<half, access::sample> h2 [[texture(10)]],
                      texture2d<half, access::sample> h3 [[texture(11)]],
                      texture1d<float, access::sample> l0 [[texture(12)]],
                      texture1d<float, access::sample> l1 [[texture(13)]],
                      texture1d<float, access::sample> l2 [[texture(14)]],
                      texture1d<half, access::sample> lh [[texture(15)]],
                      const device float2 *coord [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      uint tid [[thread_position_in_grid]])
{
    float2 p = coord[tid];
    device float4 *o = out + 16 * tid;
    o[0] = f0.sample(bicubic_pixel, p);
    o[1] = f1.sample(bicubic_pixel, p);
    o[2] = f2.sample(bicubic_pixel, p);
    o[3] = f3.sample(bicubic_pixel, p);
    o[4] = f4.sample(bicubic_pixel, p);
    o[5] = f5.sample(bicubic_pixel, p);
    o[6] = f6.sample(bicubic_pixel, p);
    o[7] = f7.sample(bicubic_pixel, p);
    o[8] = float4(h0.sample(bicubic_pixel, p));
    o[9] = float4(h1.sample(bicubic_pixel, p));
    o[10] = float4(h2.sample(bicubic_pixel, p));
    o[11] = float4(h3.sample(bicubic_pixel, p));
    o[12] = l0.sample(bicubic_pixel, p.x);
    o[13] = l1.sample(bicubic_pixel, p.x);
    o[14] = l2.sample(bicubic_pixel, p.x);
    o[15] = float4(lh.sample(bicubic_pixel, p.x));
}
