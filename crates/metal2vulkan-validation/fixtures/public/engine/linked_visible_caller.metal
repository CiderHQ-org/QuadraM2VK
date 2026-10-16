// Owned MSL: kernels that call visible functions they do not define. Each call
// is a visible-function reference (`!air.visible_function_references`) that the pipeline's
// linked functions resolve; `linked_visible_functions.metal` defines them. The module also
// defines `linked_offset`, which those functions reference in turn. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] uint linked_scale(uint x);
[[visible]] float linked_blend(float a, float b);
[[visible]] float4 linked_sample(texture2d<float> t, float2 uv);

[[visible]] uint linked_offset(uint x) { return x & 7u; }

kernel void linked_visible_caller(device uint *words [[buffer(0)]],
                                  device float *floats [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]]) {
    words[tid] = linked_scale(tid);
    floats[tid] = linked_blend(float(tid), 0.5f);
}

// Samples through the linked `linked_sample`, whose constexpr sampler the linked module states.
kernel void linked_visible_sampling(texture2d<float> t [[texture(0)]],
                                    device float4 *out [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]]) {
    out[tid] = linked_sample(t, float2(tid) * 0.25f);
}
