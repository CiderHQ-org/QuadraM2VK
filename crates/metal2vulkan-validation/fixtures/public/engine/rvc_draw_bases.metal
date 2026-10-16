// Owned fixture (written for this repository): a draw's base vertex and base
// instance. Rasterization-disabled points; vertex `vid` of instance `iid` writes `vid`, `iid`,
// `[[base_vertex]]` and `[[base_instance]]` to the four words of out[(iid % 4) * 8 + vid % 8], so
// each vertex of up to eight consecutive ones and each instance of up to four consecutive ones
// lands in a slot of its own. Metal's base vertex is the draw's `vertexStart` without an index
// buffer and its `baseVertex` with one; its base instance is the draw's `baseInstance`. The narrow
// entry reads both as `ushort`, their low 16 bits. The post-tessellation entry reads
// `[[base_instance]]`, which Metal also allows there (and refuses `[[base_vertex]]`, as it
// refuses both in mesh and object functions). Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex void draw_bases_in_a_vertex_function(uint vid [[vertex_id]], uint iid [[instance_id]],
                                            uint base_vertex [[base_vertex]],
                                            uint base_instance [[base_instance]],
                                            device uint4 *out [[buffer(0)]])
{
    out[(iid % 4) * 8 + vid % 8] = uint4(vid, iid, base_vertex, base_instance);
}

vertex void narrow_draw_bases_in_a_vertex_function(uint vid [[vertex_id]], uint iid [[instance_id]],
                                                   ushort base_vertex [[base_vertex]],
                                                   ushort base_instance [[base_instance]],
                                                   device uint4 *out [[buffer(0)]])
{
    out[(iid % 4) * 8 + vid % 8] = uint4(vid, iid, uint(base_vertex), uint(base_instance));
}

[[patch(triangle, 3)]]
vertex float4 base_instance_after_tessellation(uint base_instance [[base_instance]],
                                               uint pid [[patch_id]],
                                               float3 coord [[position_in_patch]])
{
    return float4(coord.xy, float(base_instance), float(pid) + 1.0f);
}
