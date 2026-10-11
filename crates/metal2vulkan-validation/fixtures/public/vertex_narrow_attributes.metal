#include <metal_stdlib>
using namespace metal;

struct NarrowAttributes {
    uint byte [[attribute(0)]];
    uint2 words [[attribute(1)]];
};

vertex void vertex_narrow_attributes(
    NarrowAttributes input [[stage_in]],
    device uint *output [[buffer(0)]],
    uint vertex_id [[vertex_id]])
{
    output[vertex_id] = input.byte + input.words.x + input.words.y;
}
