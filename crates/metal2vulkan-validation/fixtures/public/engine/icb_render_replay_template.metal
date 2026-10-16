// Owned fixture: the render template an executed indirect draw runs, a
// rasterization-disabled vertex function. Vertex `vid` of instance `iid` writes
// `params[0] * (vid + 1) + params[1] + 1000 * iid` to `out[16 * iid + vid]`. A command binds
// `params` and `out` (each at an offset into a larger buffer) as vertex buffers 0 and 1. A draw's
// vertex start and base instance offset `vid` and `iid`. Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex void icb_render_replay_template(
    constant uint *params [[buffer(0)]],
    device uint *out [[buffer(1)]],
    uint vid [[vertex_id]],
    uint iid [[instance_id]])
{
    out[16 * iid + vid] = params[0] * (vid + 1) + params[1] + 1000 * iid;
}
