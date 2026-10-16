// Owned fixture: how Metal rounds a `float` to `half` at the edge of the
// half range and at the half denormals, in a conversion and in `half` arithmetic, in Metal's
// compile without fast math. Thread `i` reads the float vectors (x, y, z) from `in[3i .. 3i + 2]`
// and writes 16 words to `out[4i .. 4i + 3]`, each 4-lane `half` result as two words (lane 0 in
// the low half of the first): half(x), half(y), hx + hy, hx * hy, hx - hy, fma(hx, hy, hz),
// half(x * y), and hx * hy + hz, where hx, hy and hz are half(x), half(y) and half(z). Every value
// from 65504 to below 65520 rounds to 65504 to nearest even, and from 65520 up to infinity.
// `dfl_half_edge_fast` is the same program under the default fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void dfl_half_edge(device const uint4 *in [[buffer(0)]],
                          device uint4 *out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]])
{
    float4 x = as_type<float4>(in[3 * tid]);
    float4 y = as_type<float4>(in[3 * tid + 1]);
    float4 z = as_type<float4>(in[3 * tid + 2]);
    half4 hx = half4(x), hy = half4(y), hz = half4(z);
    device uint4 *o = out + 4 * tid;
    o[0] = uint4(as_type<uint2>(hx), as_type<uint2>(hy));
    o[1] = uint4(as_type<uint2>(hx + hy), as_type<uint2>(hx * hy));
    o[2] = uint4(as_type<uint2>(hx - hy), as_type<uint2>(fma(hx, hy, hz)));
    o[3] = uint4(as_type<uint2>(half4(x * y)), as_type<uint2>(hx * hy + hz));
}
