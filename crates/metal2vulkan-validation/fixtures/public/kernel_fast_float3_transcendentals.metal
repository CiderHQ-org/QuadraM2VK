#include <metal_stdlib>
using namespace metal;

kernel void kernel_fast_float3_transcendentals(
    device float *out [[buffer(0)]],
    device const float *in [[buffer(1)]])
{
    float3 a = float3(in[0], in[1], in[2]);
    float3 b = float3(in[3], in[4], in[5]);
    float3 c = float3(in[6], in[7], in[8]);
    float3 d = float3(in[9], in[10], in[11]);
    float3 e = float3(in[12], in[13], in[14]);
    float3 l2 = fast::log2(a);
    float3 ln = fast::log(b);
    float3 e2 = fast::exp2(c);
    float3 rs = fast::rsqrt(d);
    float3 sn = fast::sin(e);
    out[0]=l2.x; out[1]=l2.y; out[2]=l2.z;
    out[3]=ln.x; out[4]=ln.y; out[5]=ln.z;
    out[6]=e2.x; out[7]=e2.y; out[8]=e2.z;
    out[9]=rs.x; out[10]=rs.y; out[11]=rs.z;
    out[12]=sn.x; out[13]=sn.y; out[14]=sn.z;
}
