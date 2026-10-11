#include <metal_stdlib>
using namespace metal;

kernel void copy_word(const device uint *input [[buffer(0)]],
                      device uint *output [[buffer(1)]]) {
    output[0] = input[0];
}
