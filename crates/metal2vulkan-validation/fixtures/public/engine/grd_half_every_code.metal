// Owned fixture: how Metal converts every code of a 16-bit normalized format to `half` when a `half`
// texture reads or samples one texel of it. Two 2D `half` textures, each read at the lane's texel
// and sampled at its centre through a nearest-filter sampler (clamp-to-edge, pixel coordinates);
// the cases bind a 16-bit normalized format to each slot, its texels holding every code once. Lane
// `i` takes texel `(i mod w, i div w)` of a `w`-wide texture and writes its four results, as `half`
// bits, to `out[4i .. 4i + 3]`: slot 0's read and sample, then slot 1's. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_pixel(filter::nearest, address::clamp_to_edge, coord::pixel);

kernel void grd_half_every_code(texture2d<half, access::sample> t0 [[texture(0)]],
                                texture2d<half, access::sample> t1 [[texture(1)]],
                                device ushort4 *out [[buffer(0)]],
                                uint tid [[thread_position_in_grid]])
{
    uint w = t0.get_width();
    uint2 texel = uint2(tid % w, tid / w);
    float2 centre = float2(texel) + 0.5;
    device ushort4 *o = out + 4 * tid;
    o[0] = as_type<ushort4>(t0.read(texel));
    o[1] = as_type<ushort4>(t0.sample(nearest_pixel, centre));
    o[2] = as_type<ushort4>(t1.read(texel));
    o[3] = as_type<ushort4>(t1.sample(nearest_pixel, centre));
}
