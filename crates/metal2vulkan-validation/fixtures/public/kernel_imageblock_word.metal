#include <metal_stdlib>
using namespace metal;

struct ImageblockWord {
    uint value;
};

kernel void imageblock_word(
    imageblock<ImageblockWord, imageblock_layout_explicit> block,
    device uint *output [[buffer(0)]],
    ushort2 thread_position [[thread_position_in_threadgroup]]) {
    threadgroup_imageblock ImageblockWord *cell = block.data(thread_position);
    cell->value = 42;
    output[0] = cell->value;
}
