// Owned fixture: the grids of mesh draws, for the runtime lane and Metal
// probe 4. Each mesh threadgroup draws one small triangle over the centre of one pixel of an 8x8
// target and gives it, as per-primitive data, the builtins its threadgroup read: with no object
// function the pixel is the threadgroup's position, and under one it is the mesh threadgroup's x
// and the object threadgroup's linear index times 2 plus the mesh threadgroup's y. The fragment
// function writes the two words to two RGBA32Uint attachments. A pair `x | y << 16` packs two
// axes. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct DrawsVertex {
    float4 position [[position]];
};

struct DrawsPrimitive {
    uint4 a [[flat]];
    uint4 b [[flat]];
};

using DrawsMesh = metal::mesh<DrawsVertex, DrawsPrimitive, 3, 1, topology::triangle>;

// A threadgroup's size packed 10 bits an axis.
static uint pack3(uint3 v) {
    return v.x | (v.y << 10) | (v.z << 20);
}

// One triangle over the centre of pixel (x, y) of the 8x8 target (row 0 at the top), and no
// other pixel's centre, with data `p`.
static void cover(DrawsMesh out, uint x, uint y, DrawsPrimitive p) {
    float q = 0.25 * (2.0 / 8.0);
    float2 c = float2((float(x) + 0.5) / 4.0 - 1.0, 1.0 - (float(y) + 0.5) / 4.0);
    DrawsVertex v;
    v.position = float4(c.x - q, c.y - q, 0.5, 1.0);
    out.set_vertex(0, v);
    v.position = float4(c.x + 3.0 * q, c.y - q, 0.5, 1.0);
    out.set_vertex(1, v);
    v.position = float4(c.x - q, c.y + 3.0 * q, 0.5, 1.0);
    out.set_vertex(2, v);
    out.set_index(0, 0);
    out.set_index(1, 1);
    out.set_index(2, 2);
    out.set_primitive(0, p);
    out.set_primitive_count(1);
}

// Without an object function: its threadgroup's position, the grid, and its threads.
[[mesh]] void mesh_draws_grid(DrawsMesh out,
                              uint tid [[thread_index_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint3 groups [[threadgroups_per_grid]],
                              uint3 tptg [[threads_per_threadgroup]],
                              uint3 tpg [[threads_per_grid]],
                              uint3 pos [[thread_position_in_grid]]) {
    if (tid == 0) {
        DrawsPrimitive p;
        p.a = uint4(group.x, group.y, groups.x, groups.y);
        p.b = uint4(tpg.x, tpg.y, pack3(tptg), pos.x | (pos.y << 16));
        cover(out, group.x, group.y, p);
    }
}

struct DrawsPayload {
    uint4 object;
    uint4 grid;
};

// Thread 0 of each object threadgroup states its own builtins in the payload and launches
// `ctl.x + group.x` by `ctl.y` mesh threadgroups.
[[object]] void mesh_draws_object(object_data DrawsPayload& payload [[payload]],
                                  mesh_grid_properties grid,
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint3 groups [[threadgroups_per_grid]],
                                  uint3 tptg [[threads_per_threadgroup]],
                                  uint3 tpg [[threads_per_grid]],
                                  constant uint4& ctl [[buffer(0)]]) {
    if (tid == 0) {
        payload.object = uint4(group.x | (group.y << 16), groups.x | (groups.y << 16),
                               pack3(tptg), tpg.x | (tpg.y << 16));
        payload.grid = uint4(group.x + groups.x * group.y, ctl.x + group.x, ctl.y, 0);
        grid.set_threadgroups_per_grid(uint3(ctl.x + group.x, ctl.y, 1));
    }
}

// Under the object function: the payload's object words, and its own position, grid and threads.
[[mesh]] void mesh_draws_mesh(DrawsMesh out,
                              const object_data DrawsPayload& payload [[payload]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint3 groups [[threadgroups_per_grid]],
                              uint3 tptg [[threads_per_threadgroup]]) {
    if (tid == 0) {
        DrawsPrimitive p;
        p.a = payload.object;
        p.b = uint4(group.x, group.y, groups.x | (groups.y << 16), pack3(tptg));
        cover(out, group.x, payload.grid.x * 2 + group.y, p);
    }
}

struct DrawsIn {
    DrawsVertex v;
    DrawsPrimitive p;
};

struct DrawsOut {
    uint4 a [[color(0)]];
    uint4 b [[color(1)]];
};

fragment DrawsOut mesh_draws_fragment(DrawsIn in [[stage_in]]) {
    DrawsOut o;
    o.a = in.p.a;
    o.b = in.p.b;
    return o;
}
