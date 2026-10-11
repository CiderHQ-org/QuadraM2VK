#include <metal_stdlib>
using namespace metal;

constant uint4 vector_value [[function_constant(0)]];

kernel void kernel_vector_function_constant(device uint *output [[buffer(0)]])
{
    output[0] = vector_value.x + vector_value.y + vector_value.z + vector_value.w;
}
