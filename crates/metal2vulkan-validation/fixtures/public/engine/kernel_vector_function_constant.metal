// Owned fixture: the compiled replacement for the hand-written top-level
// `kernel_vector_function_constant.ll`, whose function-constant global has no
// `!air.function_constants` entry, so a value supplied for index 0 names no declared constant
// (`FunctionConstantIndex`). This is the same program as that fixture's `.metal` twin. Metal
// spells a `uint4` function constant as an `air.fc_initializer` global that
// `!air.function_constants` names at index 0, plus an `air.static_init` constructor
// (`llvm.global_ctors`) that copies it into the program-scope `constant` the kernel reads.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant uint4 vector_value [[function_constant(0)]];

kernel void kernel_vector_function_constant(device uint *output [[buffer(0)]])
{
    output[0] = vector_value.x + vector_value.y + vector_value.z + vector_value.w;
}
