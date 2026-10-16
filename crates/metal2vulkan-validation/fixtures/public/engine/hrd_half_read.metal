// Owned fixture: the `half` a texture access hands the function for a
// half-float texel, every half subnormal, both zeros, both infinities, the largest half, and NaNs
// included. Lane `i` reads texel `(i mod 64, i div 64)` of a 64 x 16 texture through
// `access::sample` (a read, and a nearest sample at the texel's centre with normalized
// coordinates) and through `access::read`, and writes the three results' bits and the first read
// widened to `float` to `out[3i .. 3i + 2]`. Compiled without fast math, so a NaN, an infinity, or
// a signed zero is a defined value. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void hrd_half_read(texture2d<half, access::sample> t [[texture(0)]],
                          texture2d<half, access::read> r [[texture(1)]],
                          device uint4 *out [[buffer(0)]],
                          uint tid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    uint2 texel = uint2(tid % 64, tid / 64);
    half4 a = t.read(texel);
    half4 b = t.sample(s, (float2(texel) + 0.5f) / float2(64.0f, 16.0f));
    half4 c = r.read(texel);
    device uint4 *o = out + 3 * tid;
    o[0] = uint4(as_type<uint2>(a), as_type<uint2>(b));
    o[1] = uint4(as_type<uint2>(c), 0u, 0u);
    o[2] = as_type<uint4>(float4(a));
}
