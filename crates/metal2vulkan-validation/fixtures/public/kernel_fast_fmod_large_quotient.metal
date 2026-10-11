#include <metal_stdlib>
using namespace metal;

kernel void kernel_fast_fmod_large_quotient(
    device uint *out [[buffer(0)]],
    device const float *in [[buffer(1)]])
{
    out[0] = as_type<uint>(fast::fmod(in[0], in[1]));
    out[1] = as_type<uint>(fast::fmod(in[2], in[3]));
    out[2] = as_type<uint>(fast::fmod(in[4], in[5]));
    out[3] = as_type<uint>(fast::fmod(in[6], in[7]));
    out[4] = as_type<uint>(fast::fmod(in[8], in[9]));
    out[5] = as_type<uint>(fast::fmod(in[10], in[11]));
    out[6] = as_type<uint>(fast::fmod(in[12], in[13]));
    out[7] = as_type<uint>(fast::fmod(in[14], in[15]));

    float2 a = fast::fmod(float2(in[16], in[17]), float2(in[18], in[19]));
    float2 b = fast::fmod(float2(in[20], in[21]), float2(in[22], in[23]));
    out[8]  = as_type<uint>(a.x);
    out[9]  = as_type<uint>(a.y);
    out[10] = as_type<uint>(b.x);
    out[11] = as_type<uint>(b.y);
}
