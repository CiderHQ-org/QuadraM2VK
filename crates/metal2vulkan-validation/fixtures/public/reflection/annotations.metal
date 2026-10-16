// Owned fixture: a kernel with a user annotation
// (MTLFunctionReflection.userAnnotation, the string the attribute passes) beside one without.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[user_annotation("reflection probe annotation")]]
kernel void reflection_annotated(device uint *out [[buffer(0)]],
                                 uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}

kernel void reflection_unannotated(device uint *out [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]]) {
    out[tid] = tid + 1u;
}
