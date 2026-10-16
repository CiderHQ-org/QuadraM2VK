// Owned fixture: a rasterization-disabled vertex function whose device
// argument buffer holds a render pipeline state, a compute pipeline state, a `scale`, and a
// `bias` (at 0, 8, 16, and 20). Each vertex writes `vid * scale + bias` and never reads a
// pipeline state. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Pipelines {
    render_pipeline_state render;
    compute_pipeline_state compute;
    uint scale;
    uint bias;
};

vertex void k14_pipeline_state_vertex(
    device const Pipelines &pipelines [[buffer(0)]],
    device uint *out [[buffer(1)]],
    uint vid [[vertex_id]])
{
    out[vid] = vid * pipelines.scale + pipelines.bias;
}
