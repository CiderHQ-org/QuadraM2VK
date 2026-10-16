// Owned fixture: the render template of executed draws whose vertex
// buffer a command binds at an attribute stride, a rasterization-disabled vertex function. Vertex
// `vid` of instance `iid` fetches its `[[stage_in]]` attribute (a case's vertex descriptor: one
// uint per vertex, under a layout of `MTLBufferLayoutStrideDynamic` the stride the buffer is
// bound with) and writes it with its vertex and instance ids:
//     out[vid] = (v, vid | iid << 16)
// A command binds `out` as vertex buffer 0, and the attribute's data at the slot the case's
// vertex descriptor gives it. A draw's vertex start offsets `vid`. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    uint v [[attribute(0)]];
};

vertex void icb_render_stride_template(
    Vertex in [[stage_in]],
    device uint2 *out [[buffer(0)]],
    uint vid [[vertex_id]],
    uint iid [[instance_id]])
{
    out[vid] = uint2(in.v, vid | (iid << 16));
}
