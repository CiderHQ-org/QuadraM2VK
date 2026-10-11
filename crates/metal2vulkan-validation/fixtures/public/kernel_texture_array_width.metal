#include <metal_stdlib>
using namespace metal;

kernel void texture_array_width(
    array<texture2d<float, access::sample>, 2> textures [[texture(0)]],
    device uint *out [[buffer(0)]]) {
    out[0] = textures[1].get_width();
}
