// Owned fixture: every mesh output and object input the AIR decode
// reads, compiled by Apple's compiler: points with point_size, clip distances and user data of
// several types; primitives with primitive_culled, primitive_id, render_target_array_index,
// viewport_array_index and user data; lines without primitive data, written through set_indices;
// an object function with threadgroup memory, a payload through a pointer and its stage
// attributes; a fragment function reading primitive_id and point_coord. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct MeshOutputsVertex {
    float4 position [[position]];
    float size [[point_size]];
    float clip [[clip_distance]] [2];
    half2 uv [[user(uvs)]];
    int k [[flat]];
    float c [[center_no_perspective]];
};

struct MeshOutputsPrimitive {
    bool culled [[primitive_culled]];
    uint pid [[primitive_id]];
    uint layer [[render_target_array_index]];
    uint viewport [[viewport_array_index]];
    float2 pc;
};

using MeshOutputsPoints = metal::mesh<MeshOutputsVertex, MeshOutputsPrimitive, 16, 8, topology::point>;

[[mesh]] void mesh_outputs_points(MeshOutputsPoints out,
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint3 tpt [[thread_position_in_threadgroup]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint3 tptg [[threads_per_threadgroup]],
                                  uint3 groups [[threadgroups_per_grid]],
                                  uint3 tpig [[thread_position_in_grid]],
                                  uint sgi [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint width [[threads_per_simdgroup]],
                                  constant uint& n [[buffer(1)]]) {
    threadgroup float scratch[64];
    scratch[tid & 63u] = float(tpt.x + group.y + tptg.z + groups.x + tpig.y + sgi + lane + width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        MeshOutputsVertex v;
        v.position = float4(scratch[tid]);
        v.size = 3;
        v.clip[0] = 1;
        v.clip[1] = 2;
        v.uv = half2(1);
        v.k = int(tid);
        v.c = 0.5;
        out.set_vertex(tid, v);
    }
    if (tid < 8) {
        out.set_index(tid, tid);
        MeshOutputsPrimitive p;
        p.culled = (tid & 1u) != 0;
        p.pid = tid;
        p.layer = 0;
        p.viewport = 0;
        p.pc = float2(tid);
        out.set_primitive(tid, p);
    }
    out.set_primitive_count(n);
}

struct MeshOutputsLineVertex {
    float4 position [[position, invariant]];
};

using MeshOutputsLines = metal::mesh<MeshOutputsLineVertex, void, 8, 4, topology::line>;

[[mesh, max_total_threads_per_threadgroup(32)]]
void mesh_outputs_lines(MeshOutputsLines out, uint tid [[thread_index_in_threadgroup]]) {
    MeshOutputsLineVertex v;
    v.position = float4(float(tid));
    out.set_vertex(tid & 7u, v);
    if (tid < 2) {
        out.set_indices(tid * 4, uchar4(0, 1, 2, 3));
    }
    if (tid == 2) {
        out.set_indices(6, uchar2(4, 5));
    }
    out.set_primitive_count(4);
}

struct MeshOutputsPointIn {
    float4 position [[position]];
    half2 uv [[user(uvs)]];
    int k [[flat]];
    float c [[center_no_perspective]];
};

struct MeshOutputsPrimitiveIn {
    float2 pc;
};

struct MeshOutputsFragmentIn {
    MeshOutputsPointIn v;
    MeshOutputsPrimitiveIn p;
};

fragment float4 mesh_outputs_fragment(MeshOutputsFragmentIn in [[stage_in]],
                                      uint prim [[primitive_id]],
                                      float2 coord [[point_coord]]) {
    return float4(float(in.v.uv.x), float(in.v.k), in.p.pc.x + float(prim), coord.x + in.v.c);
}

struct MeshOutputsPayload {
    float4 a;
};

[[object, max_total_threads_per_threadgroup(64), max_total_threadgroups_per_mesh_grid(16)]]
void mesh_outputs_object(object_data MeshOutputsPayload* payload [[payload]],
                         mesh_grid_properties grid,
                         uint tid [[thread_index_in_threadgroup]],
                         uint3 tptg [[threads_per_threadgroup]],
                         uint3 tpig [[thread_position_in_grid]],
                         uint3 groups [[threadgroups_per_grid]],
                         threadgroup uint* shared [[threadgroup(0)]]) {
    shared[tid] = tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    payload[tid & 3u].a = float4(float(shared[tid ^ 1u] + tpig.x + groups.y));
    if (tid == 1) {
        grid.set_threadgroups_per_grid(uint3(tptg.x, 1, 1));
    }
}

using MeshOutputsTriangles = metal::mesh<MeshOutputsLineVertex, MeshOutputsPrimitiveIn, 8, 4, topology::triangle>;

[[mesh, max_total_threads_per_threadgroup(32)]]
void mesh_outputs_triangles(MeshOutputsTriangles out,
                            const object_data MeshOutputsPayload* payload [[payload]],
                            uint tid [[thread_index_in_threadgroup]]) {
    MeshOutputsLineVertex v;
    v.position = payload[tid & 3u].a;
    out.set_vertex(tid & 7u, v);
    if (tid < 3) {
        out.set_indices(tid * 4, uchar4(1, 2, 3, 4));
        MeshOutputsPrimitiveIn p;
        p.pc = float2(1.0);
        out.set_primitive(tid, p);
    }
    out.set_primitive_count(3);
}
