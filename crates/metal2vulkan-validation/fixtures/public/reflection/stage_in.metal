// Owned fixture: a kernel's stage input, for which Metal's
// pipeline reflection lists the stage input descriptor's buffers (`stageInBuffer.N`) and the
// function its stage input attributes. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ReflectionKernelIn {
    float4 a [[attribute(0)]];
    uint b [[attribute(1)]];
};

kernel void reflection_stage_in(ReflectionKernelIn in [[stage_in]],
                                device float4 *out [[buffer(0)]],
                                uint tid [[thread_position_in_grid]]) {
    out[tid] = in.a + float(in.b);
}
