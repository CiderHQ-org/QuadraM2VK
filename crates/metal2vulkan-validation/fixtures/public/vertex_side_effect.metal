#include <metal_stdlib>
using namespace metal;

vertex void vertex_side_effect(
    device uint *output [[buffer(0)]],
    uint vertex_id [[vertex_id]])
{
    output[vertex_id] = 42;
}
