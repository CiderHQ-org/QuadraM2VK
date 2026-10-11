#include <metal_stdlib>
using namespace metal;

kernel void kernel_unpack_packed_pixels(
    device float *out [[buffer(0)]],
    device const uint *in [[buffer(1)]],
    device const rg11b10f<float3> *fpix [[buffer(2)]],
    device const rgb9e5<float3> *epix [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    float4 a = unpack_unorm10a2_to_float(in[0]);
    float3 b = fpix[0];
    float3 c = epix[0];
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z;
    out[7] = c.x; out[8] = c.y; out[9] = c.z;
}
