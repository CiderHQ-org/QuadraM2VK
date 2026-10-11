#include <metal_stdlib>
using namespace metal;

struct IBColor {
    half4 value [[color(0)]];
};

kernel void kernel_imageblock_extent(
    imageblock<IBColor, imageblock_layout_implicit> block,
    device uint *out [[buffer(0)]],
    ushort2 pos [[thread_position_in_threadgroup]])
{
    uint idx = uint(pos.y) * 16u + uint(pos.x);
    uint o = idx * 4;
    out[o + 0] = uint(block.get_width());
    out[o + 1] = uint(block.get_height());
    out[o + 2] = uint(as_type<ushort>(block.read(pos).value.x));
    out[o + 3] = uint(as_type<ushort>(block.read(pos).value.z));
}
