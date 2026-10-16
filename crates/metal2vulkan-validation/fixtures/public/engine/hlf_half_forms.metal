// Owned fixture: how Metal converts texels to `half` in the sampling forms
// `bcx_half_read` does not read. Four 2D `half` textures (the cases bind a 32-bit float, a 16-bit
// unorm, an 8-bit unorm, and an sRGB format) are each gathered (components x and w) and sampled
// with normalized coordinates, clamp-to-edge: nearest, and linear with a constant offset, an
// explicit level 0, a bias, and zero gradients. Lane `i` reads the pixel coordinate `coord[i]`,
// divides it by the texture's size, and writes seven results per texture, widened to `float`, to
// `out[28i ..]` in slot order. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_edge(filter::nearest, address::clamp_to_edge);
constant sampler linear_edge(filter::linear, address::clamp_to_edge);

// One texture's seven forms, written to o[0 .. 6] (a macro, so the entry stays one function).
#define FORMS(t, o)                                                                             \
    do {                                                                                        \
        float2 u = p / float2(t.get_width(), t.get_height());                                  \
        (o)[0] = float4(t.gather(nearest_edge, u, int2(0), component::x));                      \
        (o)[1] = float4(t.gather(nearest_edge, u, int2(0), component::w));                      \
        (o)[2] = float4(t.sample(nearest_edge, u));                                             \
        (o)[3] = float4(t.sample(linear_edge, u, int2(1, -1)));                                 \
        (o)[4] = float4(t.sample(linear_edge, u, level(0.0)));                                  \
        (o)[5] = float4(t.sample(linear_edge, u, bias(0.75)));                                  \
        (o)[6] = float4(t.sample(linear_edge, u, gradient2d(float2(0.0), float2(0.0))));        \
    } while (false)

kernel void hlf_half_forms(texture2d<half, access::sample> t0 [[texture(0)]],
                           texture2d<half, access::sample> t1 [[texture(1)]],
                           texture2d<half, access::sample> t2 [[texture(2)]],
                           texture2d<half, access::sample> t3 [[texture(3)]],
                           const device float2 *coord [[buffer(0)]],
                           device float4 *out [[buffer(1)]],
                           uint tid [[thread_position_in_grid]])
{
    float2 p = coord[tid];
    device float4 *o = out + 28 * tid;
    FORMS(t0, o);
    FORMS(t1, o + 7);
    FORMS(t2, o + 14);
    FORMS(t3, o + 21);
}
