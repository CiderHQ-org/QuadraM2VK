#include <metal_stdlib>
using namespace metal;

struct StageInputWord {
    uint3 value [[attribute(0)]];
};

kernel void stage_input_word(StageInputWord input [[stage_in]],
                             device uint *output [[buffer(0)]]) {
    output[0] = input.value.x;
}
