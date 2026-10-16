// Owned fixture: `dfl_half_edge`'s program under the default fast math:
// how Metal rounds a `float` to `half` at the edge of the half range and at the half denormals, in
// a conversion and in `half` arithmetic carrying every flag. Thread `i` reads (x, y, z) and writes
// 16 words, `dfl_half_edge`'s rows. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void dfl_half_edge_fast(device const uint4 *in [[buffer(0)]],
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
