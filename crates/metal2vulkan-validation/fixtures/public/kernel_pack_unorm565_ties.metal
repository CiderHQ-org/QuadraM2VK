#include <metal_stdlib>
using namespace metal;

kernel void kernel_pack_unorm565_ties(
    device const float4  *fin [[buffer(0)]],
    device const ushort4 *hin [[buffer(1)]],
    device uint          *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    float4 f = fin[tid];
    half4  h = as_type<half4>(hin[tid]);

    uint b = tid * 2;
    out[b + 0] = uint(pack_float_to_unorm565(f.xyz));
    out[b + 1] = uint(pack_half_to_unorm565(h.xyz));
}
