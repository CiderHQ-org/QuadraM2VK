// Owned fixture: two half taps of a texture at a coordinate, then a step of the coordinate, then
// the taps' maximum. Thread t reads its coordinate c from `coords[t]`, takes the texels at c and
// one to its right through a nearest, pixel-coordinate sampler, steps c.x by `step` and stores c
// back, and writes (max(a.rg, b.rg), 0, 1) to texel (t, 0) of `dst`. SPIRV-Cross (MoltenVK 1.4.1)
// forwards each tap's `OpQuantizeToF16` into the maximum without the tap's dependency on c, and
// steps c in place: without the translator's copies the taps read the stepped c. Loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler nearest(coord::pixel, filter::nearest, address::clamp_to_edge);

kernel void sxc_taps_before_step(texture2d<half> src [[texture(0)]],
                                 texture2d<half, access::write> dst [[texture(1)]],
                                 device float2 *coords [[buffer(0)]],
                                 constant float &step [[buffer(1)]],
                                 uint t [[thread_position_in_grid]]) {
    float2 c = coords[t];
    half4 a = src.sample(nearest, c);
    half4 b = src.sample(nearest, c, int2(1, 0));
    c.x += step;
    coords[t] = c;
    dst.write(half4(max(a.rg, b.rg), 0.0h, 1.0h), uint2(t, 0));
}
