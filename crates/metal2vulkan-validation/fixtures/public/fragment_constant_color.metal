#include <metal_stdlib>
using namespace metal;

fragment float4 fragment_constant_color() {
    return float4(0.25, 0.5, 0.75, 1.0);
}
