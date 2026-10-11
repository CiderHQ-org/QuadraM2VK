#include <metal_stdlib>
using namespace metal;

kernel void bounded_loop_word(device uint *output [[buffer(0)]],
                              const device uint *count [[buffer(1)]]) {
    uint sum = 0;
    for (uint index = 0; index < count[0]; ++index) {
        sum += index + 1;
    }
    output[0] = sum;
}
