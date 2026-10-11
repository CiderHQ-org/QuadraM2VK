#include <metal_stdlib>
using namespace metal;

using WordFunction = uint(uint);

kernel void kernel_visible_function_table_word(
    device uint *output [[buffer(0)]],
    visible_function_table<WordFunction> functions [[buffer(1)]]) {
    output[0] = functions[0](41u);
}
