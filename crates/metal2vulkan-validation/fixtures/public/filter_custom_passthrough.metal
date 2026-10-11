#include <metal_stdlib>
using namespace metal;

[[visible]] half4 custom_fn(
    float2,
    half4 color,
    constant uchar *,
    device uchar *,
    texture2d<half, access::sample>,
    texture2d<half, access::sample>,
    texture2d<half, access::sample>) {
    return color;
}
