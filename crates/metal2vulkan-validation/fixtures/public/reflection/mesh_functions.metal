// Owned fixture: an object function that writes a
// payload and a mesh function that reads it, with a buffer each, and a fragment function, for
// their library reflection (the payload binding) and a mesh render pipeline's (object, mesh and
// fragment bindings). Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ReflectionMeshPayload {
    float4 tint;
    uint count;
};

[[object]]
void reflection_object(object_data ReflectionMeshPayload &payload [[payload]],
                       mesh_grid_properties grid,
                       constant float4 &tint [[buffer(0)]],
                       uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        payload.tint = tint;
        payload.count = 1;
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

struct ReflectionMeshVertex {
    float4 position [[position]];
    float4 colour;
};

using ReflectionMesh = mesh<ReflectionMeshVertex, void, 3, 1, topology::triangle>;

[[mesh]]
void reflection_mesh(ReflectionMesh m,
                     const object_data ReflectionMeshPayload &payload [[payload]],
                     constant float2 *corners [[buffer(1)]],
                     uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        m.set_primitive_count(payload.count);
    }
    if (tid < 3) {
        ReflectionMeshVertex v;
        v.position = float4(corners[tid], 0.0, 1.0);
        v.colour = payload.tint;
        m.set_vertex(tid, v);
        m.set_index(tid, uchar(tid));
    }
}

fragment float4 reflection_mesh_fragment(ReflectionMeshVertex in [[stage_in]]) { return in.colour; }
