// Owned fixture: vertex attribute fetches past their vertex buffer, the
// vertex input containment form. A rasterization-disabled point per vertex; vertex v writes the
// attribute it fetched to out[v]. The case's vertex buffer holds four elements and the draw has
// eight vertices, so vertices 4 to 7 fetch past it. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct In {
    uint v [[attribute(0)]];
};

vertex void ob3_vertex_input(In in [[stage_in]],
                             device uint *out [[buffer(1)]],
                             uint vid [[vertex_id]]) {
    out[vid] = in.v;
}
