// Owned fixture: how Metal converts texels to `half` under the address
// modes `bcx_half_read` does not read. Four 2D `half` textures (the cases bind a 32-bit float, a
// 16-bit unorm, an 8-bit unorm, and an sRGB format) are each sampled through linear samplers with
// normalized coordinates: repeat (the coordinate plus (1, -2)), mirrored repeat (the negated
// coordinate), and clamp-to-zero (the pixel coordinate times 1.125 minus 1, over the size, so the
// footprints of the first and last texels take the border). Lane `i` reads the pixel coordinate
// `coord[i]` and writes three results per texture, widened to `float`, to `out[12i ..]` in slot
// order. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler linear_repeat(filter::linear, address::repeat);
constant sampler linear_mirror(filter::linear, address::mirrored_repeat);
constant sampler linear_zero(filter::linear, address::clamp_to_zero);

// One texture's three forms, written to o[0 .. 2] (a macro, so the entry stays one function).
#define ADDRESS(t, o)                                                                           \
    do {                                                                                        \
        float2 size = float2(t.get_width(), t.get_height());                                   \
        float2 u = p / size;                                                                    \
        (o)[0] = float4(t.sample(linear_repeat, u + float2(1.0, -2.0)));                        \
        (o)[1] = float4(t.sample(linear_mirror, -u));                                           \
        (o)[2] = float4(t.sample(linear_zero, (p * 1.125 - 1.0) / size));                      \
    } while (false)

kernel void hlf_half_address(texture2d<half, access::sample> t0 [[texture(0)]],
                             texture2d<half, access::sample> t1 [[texture(1)]],
                             texture2d<half, access::sample> t2 [[texture(2)]],
                             texture2d<half, access::sample> t3 [[texture(3)]],
                             const device float2 *coord [[buffer(0)]],
                             device float4 *out [[buffer(1)]],
                             uint tid [[thread_position_in_grid]])
{
    float2 p = coord[tid];
    device float4 *o = out + 12 * tid;
    ADDRESS(t0, o);
    ADDRESS(t1, o + 3);
    ADDRESS(t2, o + 6);
    ADDRESS(t3, o + 9);
}
