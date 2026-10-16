// Owned fixture: mesh type members under [[function_constant(..)]],
// as Apple's VFX particle meshes declare theirs: a vertex data member and a primitive data member
// exist only where `has_tint` holds, beside members that always exist. Each is written only under
// its constant. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool has_tint [[function_constant(0)]];

struct MeshGatedVertex {
    float4 position [[position]];
    float2 uv;
    half4 tint [[function_constant(has_tint)]];
};

struct MeshGatedPrimitive {
    uint id [[flat]];
    float weight [[function_constant(has_tint)]];
};

using MeshGatedMesh = metal::mesh<MeshGatedVertex, MeshGatedPrimitive, 3, 1, topology::triangle>;

[[mesh]] void mesh_gated(MeshGatedMesh out, uint tid [[thread_index_in_threadgroup]]) {
    if (tid < 3) {
        MeshGatedVertex v;
        v.position = float4(float(tid & 1u), float(tid >> 1), 0.5, 1.0);
        v.uv = float2(float(tid));
        if (has_tint) {
            v.tint = half4(1.0h);
        }
        out.set_vertex(tid, v);
        out.set_index(tid, uchar(tid));
    }
    if (tid == 0) {
        MeshGatedPrimitive p;
        p.id = 7;
        if (has_tint) {
            p.weight = 0.5;
        }
        out.set_primitive(0, p);
        out.set_primitive_count(1);
    }
}
