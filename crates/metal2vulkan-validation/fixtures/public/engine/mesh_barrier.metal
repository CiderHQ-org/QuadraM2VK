// Owned fixture: mesh draws that read, behind a memory
// barrier, what an earlier draw of the same render pass wrote, for the runtime lane and Metal
// probe 5. A writer draw covers pixel (7, 7) of an 8x8 RGBA32Uint target and stores a control
// word to buffer 0, from its mesh function (`mesh_barrier_writer`) or from its fragment function
// (`mesh_barrier_fragment_writer` behind `mesh_barrier_cover`). A reader draw reads the control
// word: its object function launches `ctl.x` by `ctl.y` mesh threadgroups and passes `ctl` in
// its payload (`mesh_barrier_object` with `mesh_barrier_mesh`), or, without one, each mesh
// threadgroup of its grid below `ctl.x` by `ctl.y` draws (`mesh_barrier_grid`). Each drawing
// mesh threadgroup covers the pixel of its position with `(x, y, ctl.z, ctl.w)`. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct BarrierVertex {
    float4 position [[position]];
};

struct BarrierPrimitive {
    uint4 a [[flat]];
};

using BarrierMesh = metal::mesh<BarrierVertex, BarrierPrimitive, 3, 1, topology::triangle>;

// One triangle over the centre of pixel (x, y) of the 8x8 target (row 0 at the top), and no
// other pixel's centre, with data `a`.
static void cover(BarrierMesh out, uint x, uint y, uint4 a) {
    float q = 0.25 * (2.0 / 8.0);
    float2 c = float2((float(x) + 0.5) / 4.0 - 1.0, 1.0 - (float(y) + 0.5) / 4.0);
    BarrierVertex v;
    v.position = float4(c.x - q, c.y - q, 0.5, 1.0);
    out.set_vertex(0, v);
    v.position = float4(c.x + 3.0 * q, c.y - q, 0.5, 1.0);
    out.set_vertex(1, v);
    v.position = float4(c.x - q, c.y + 3.0 * q, 0.5, 1.0);
    out.set_vertex(2, v);
    out.set_index(0, 0);
    out.set_index(1, 1);
    out.set_index(2, 2);
    BarrierPrimitive p;
    p.a = a;
    out.set_primitive(0, p);
    out.set_primitive_count(1);
}

// The mesh-stage writer: stores (3, 2, 1, 7) and covers pixel (7, 7).
[[mesh]] void mesh_barrier_writer(BarrierMesh out,
                                  device uint4& ctl [[buffer(0)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        ctl = uint4(3, 2, 1, 7);
        cover(out, 7, 7, uint4(0xAAAA, 0, 0, 0));
    }
}

// The fragment writer's mesh function: covers pixel (7, 7), writing nothing.
[[mesh]] void mesh_barrier_cover(BarrierMesh out, uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        cover(out, 7, 7, uint4(0xBBBB, 0, 0, 0));
    }
}

struct BarrierPayload {
    uint4 ctl;
};

// The reader's object function: launches `ctl.x` by `ctl.y` mesh threadgroups.
[[object]] void mesh_barrier_object(object_data BarrierPayload& payload [[payload]],
                                    mesh_grid_properties grid,
                                    uint tid [[thread_index_in_threadgroup]],
                                    device const uint4& ctl [[buffer(0)]]) {
    if (tid == 0) {
        uint4 c = ctl;
        payload.ctl = c;
        grid.set_threadgroups_per_grid(uint3(c.x, c.y, 1));
    }
}

// The reader's mesh function under the object function.
[[mesh]] void mesh_barrier_mesh(BarrierMesh out,
                                const object_data BarrierPayload& payload [[payload]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    if (tid == 0) {
        cover(out, group.x, group.y, uint4(group.x, group.y, payload.ctl.z, payload.ctl.w));
    }
}

// The reader without an object function: its threadgroups below `ctl.x` by `ctl.y` draw.
[[mesh]] void mesh_barrier_grid(BarrierMesh out,
                                device const uint4& ctl [[buffer(0)]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    if (tid == 0) {
        uint4 c = ctl;
        if (group.x < c.x && group.y < c.y) {
            cover(out, group.x, group.y, uint4(group.x, group.y, c.z, c.w));
        }
    }
}

struct BarrierIn {
    BarrierVertex v;
    BarrierPrimitive p;
};

fragment uint4 mesh_barrier_fragment(BarrierIn in [[stage_in]]) {
    return in.p.a;
}

// The fragment-stage writer: stores (3, 2, 1, 9) from the one fragment of pixel (7, 7).
fragment uint4 mesh_barrier_fragment_writer(BarrierIn in [[stage_in]],
                                            device uint4& ctl [[buffer(0)]]) {
    ctl = uint4(3, 2, 1, 9);
    return in.p.a;
}
