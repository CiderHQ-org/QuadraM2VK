// Owned MSL: visible functions a pipeline links. `linked_scale` references
// `linked_offset`, which `linked_visible_caller.metal` defines, so linking resolves references of
// linked functions in turn; `linked_sample` samples through a constexpr sampler. The kernel only
// gives the module an entry point. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] uint linked_offset(uint x);

[[visible]] uint linked_scale(uint x) { return x * 3u + linked_offset(x); }

[[visible]] float linked_blend(float a, float b) { return a * b + 1.0f; }

// A constexpr sampler inside a linked function: its `!air.sampler_states` entry links with it.
[[visible]] float4 linked_sample(texture2d<float> t, float2 uv) {
    constexpr sampler repeat_linear(filter::linear, address::repeat);
    return t.sample(repeat_linear, uv, level(0.0));
}

kernel void linked_visible_functions(device uint *out [[buffer(0)]],
                                     uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
