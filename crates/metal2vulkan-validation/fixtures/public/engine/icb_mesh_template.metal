// Owned fixture: the rasterization-disabled mesh render pipelines an executed indirect command
// buffer's mesh draws run (the kernels that encode them are `icb_mesh_encode`'s), whose draws write
// what they ran into a device buffer. `icb_mesh_marks`' thread 0 of each mesh threadgroup writes
// its position, the grid's threadgroups and its threads, one byte per axis, and 0xd0000000 plus
// its index into the word of its index in the grid, of mesh buffer 0. `icb_object_reader` reads
// word 0 of object buffer 0, which a writer stored, and writes it doubled plus one into word 1,
// launching no mesh threadgroup of its partner `icb_mesh_triangle`, which, where a grid launches
// it, draws one triangle whose corners its payload's word places. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct MarksVertex {
    float4 position [[position]];
};

using MarksMesh = metal::mesh<MarksVertex, void, 3, 1, topology::triangle>;

// A grid position or size, one byte per axis.
static uint packed(uint3 v) {
    return v.x | (v.y << 8) | (v.z << 16);
}

[[mesh]] void icb_mesh_marks(MarksMesh out,
                             device uint4* marks [[buffer(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint3 groups [[threadgroups_per_grid]],
                             uint3 threads [[threads_per_threadgroup]]) {
    if (tid == 0) {
        uint at = group.x + groups.x * (group.y + groups.y * group.z);
        marks[at] = uint4(packed(group), packed(groups), packed(threads), 0xd0000000u | at);
        out.set_primitive_count(0);
    }
}

struct ReaderPayload {
    uint word;
};

[[object]] void icb_object_reader(object_data ReaderPayload& payload [[payload]],
                                  mesh_grid_properties grid,
                                  device uint4* ctl [[buffer(0)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        uint4 c = ctl[0];
        ctl[1] = c * 2 + 1;
        payload.word = c.x;
        grid.set_threadgroups_per_grid(uint3(0, 1, 1));
    }
}

[[mesh]] void icb_mesh_triangle(MarksMesh out,
                                const object_data ReaderPayload& payload [[payload]],
                                uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        float shift = float(payload.word & 3u) * 0.25f;
        out.set_vertex(0, MarksVertex{float4(-1.0f + shift, -1.0f, 0.0f, 1.0f)});
        out.set_vertex(1, MarksVertex{float4(3.0f, -1.0f, 0.0f, 1.0f)});
        out.set_vertex(2, MarksVertex{float4(-1.0f + shift, 3.0f, 0.0f, 1.0f)});
        out.set_index(0, 0);
        out.set_index(1, 1);
        out.set_index(2, 2);
        out.set_primitive_count(1);
    }
}
