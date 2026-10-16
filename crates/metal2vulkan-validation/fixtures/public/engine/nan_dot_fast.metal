// Owned fixture: `nan_dot`'s program under the
// default fast math, its flagged state: how Metal sums `dot` on `float` and `half` vectors of two,
// three and four lanes where the calls carry every fast-math flag. Thread `i` reads x and y from
// `in[2i]` and `in[2i + 1]` and writes `nan_dot`'s 8 words. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void nan_dot_fast(device const float4 *in [[buffer(0)]],
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
