#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

kernel void primitive_as_word(primitive_acceleration_structure acceleration_structure [[buffer(5)]],
                              device uint *output [[buffer(0)]]) {
    output[0] = 42;
}
