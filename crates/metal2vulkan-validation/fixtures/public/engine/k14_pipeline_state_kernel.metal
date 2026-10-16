// Owned fixture: a kernel whose constant argument buffer holds two
// stages, each a render pipeline state, a compute pipeline state, and a `scale`, then a `bias`
// (Metal's layout: the stages at 0 and 24, `bias` at 48). Each thread writes
// `tid * stages[tid & 1].scale + bias`: it reads only the data members, at a dynamic index over
// the stages, and never a pipeline state. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Stage {
    render_pipeline_state render;
    compute_pipeline_state compute;
    uint scale;
};

struct Encoders {
    Stage stages[2];
    uint bias;
};

kernel void k14_pipeline_state_kernel(
    constant Encoders &encoders [[buffer(0)]],
    device uint *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = tid * encoders.stages[tid & 1].scale + encoders.bias;
}
