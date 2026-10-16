// Owned fixture: index fetches past the index buffer of an indexed draw,
// the index buffer containment form. A rasterization-disabled point per index; the vertex
// function marks the vertex id it was given, seen[min(vid, 7)] = 1 (idempotent, so the marks do
// not depend on how often an index is shaded). The case's index buffer holds indices 0 to 3 and
// the draw reads more indices than it holds, so any vertex id of 4 or more comes from an index
// read past the buffer. Fetches no vertex attribute. Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex void ob3_index_buffer(device uint *seen [[buffer(0)]],
                             uint vid [[vertex_id]]) {
    seen[min(vid, 7u)] = 1u;
}
