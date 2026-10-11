#include <metal_stdlib>
using namespace metal;

kernel void kernel_srgb_unpack_all_bytes(
    device float *out [[buffer(0)]],
    device const uint *in [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float4 v = unpack_unorm4x8_srgb_to_float(in[tid]);
    out[tid * 4 + 0] = v.x;
    out[tid * 4 + 1] = v.y;
    out[tid * 4 + 2] = v.z;
    out[tid * 4 + 3] = v.w;
}
