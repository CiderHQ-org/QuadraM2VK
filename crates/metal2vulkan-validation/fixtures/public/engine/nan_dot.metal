// Owned fixture: how Metal sums `dot` on `float`
// and `half` vectors of two, three and four lanes, in Metal's compile without fast math. Thread
// `i` reads the float vectors x and y from `in[2i]` and `in[2i + 1]` and writes 8 words to
// `out[2i]` and `out[2i + 1]`: dot(x.xy, y.xy), dot(x.xyz, y.xyz), dot(x, y); the same three on
// `half(x)` and `half(y)`, each a `half`'s bits in a word; and, with a = x.x, b = x.y, c = x.z,
// dot(float2(a, b), float2(b, c)) and dot(float2(a, b), float2(b, a)), whose shared operand a
// compiler may factor out. `nan_dot_fast` is the same program under the default fast math.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_dot(device const float4 *in [[buffer(0)]],
                    device uint4 *out [[buffer(1)]],
                    uint tid [[thread_position_in_grid]])
{
    float4 x = in[2 * tid];
    float4 y = in[2 * tid + 1];
    half4 hx = half4(x);
    half4 hy = half4(y);
    out[2 * tid] = uint4(as_type<uint>(dot(x.xy, y.xy)), as_type<uint>(dot(x.xyz, y.xyz)),
                         as_type<uint>(dot(x, y)), uint(as_type<ushort>(dot(hx.xy, hy.xy))));
    out[2 * tid + 1] = uint4(uint(as_type<ushort>(dot(hx.xyz, hy.xyz))),
                             uint(as_type<ushort>(dot(hx, hy))),
                             as_type<uint>(dot(float2(x.x, x.y), float2(x.y, x.z))),
                             as_type<uint>(dot(float2(x.x, x.y), float2(x.y, x.x))));
}
