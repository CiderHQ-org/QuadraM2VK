#include <metal_stdlib>
using namespace metal;

kernel void kernel_half_transcendentals(
    device half *out [[buffer(0)]],
    device const half *in [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    half4 a = half4(in[0], in[1], in[2], in[3]);
    half4 b = half4(in[4], in[5], in[6], in[7]);
    half4 c = half4(in[8], in[9], in[10], in[11]);
    half4 l2 = log2(a);
    half4 ln = log(b);
    half4 e  = exp(c);
    out[0]=l2.x; out[1]=l2.y; out[2]=l2.z; out[3]=l2.w;
    out[4]=ln.x; out[5]=ln.y; out[6]=ln.z; out[7]=ln.w;
    out[8]=e.x;  out[9]=e.y;  out[10]=e.z; out[11]=e.w;
}
