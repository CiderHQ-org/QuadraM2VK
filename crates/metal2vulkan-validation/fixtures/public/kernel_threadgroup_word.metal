#include <metal_stdlib>
using namespace metal;

kernel void threadgroup_word(threadgroup uint *scratch [[threadgroup(0)]],
                             device uint *output [[buffer(1)]]) {
    scratch[0] = 42;
    output[0] = scratch[0];
}
