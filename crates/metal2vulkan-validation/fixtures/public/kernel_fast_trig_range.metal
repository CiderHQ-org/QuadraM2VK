#include <metal_stdlib>
using namespace metal;

kernel void kernel_fast_trig_range(
    device float *out [[buffer(0)]],
    device const float *in [[buffer(1)]])
{
    float3 a = float3(in[0], in[1], in[2]);
    float3 b = float3(in[3], in[4], in[5]);
    float3 s3 = fast::sin(a);
    float3 c3 = fast::cos(b);
    float s1 = fast::sin(in[6]);
    float c1 = fast::cos(in[7]);
    out[0]=s3.x; out[1]=s3.y; out[2]=s3.z;
    out[3]=c3.x; out[4]=c3.y; out[5]=c3.z;
    out[6]=s1; out[7]=c1;
}
