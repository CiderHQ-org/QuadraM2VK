#include <metal_stdlib>
using namespace metal;

kernel void invert_word(device uint *value [[buffer(0)]]) {
    value[0] = ~value[0];
}
