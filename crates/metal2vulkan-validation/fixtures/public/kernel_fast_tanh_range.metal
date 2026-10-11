#include <metal_stdlib>
using namespace metal;

kernel void kernel_fast_tanh_range(
    device float *out [[buffer(0)]],
    device const float *in [[buffer(1)]])
{
    float4 a = float4(in[0], in[1], in[2], in[3]);
    float4 b = float4(in[4], in[5], in[6], in[7]);
    float4 ta = fast::tanh(a);
    float4 tb = fast::tanh(b);
    out[0]=ta.x; out[1]=ta.y; out[2]=ta.z; out[3]=ta.w;
    out[4]=tb.x; out[5]=tb.y; out[6]=tb.z; out[7]=tb.w;
}
