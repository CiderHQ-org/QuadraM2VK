// Owned fixture: what Apple's VFX particle meshes read and write beside
// their user data. A mesh function reads its vertex amplification's id and count (one entry), and
// reads the id under [[function_constant(..)]] (another); a third writes each primitive's render
// target slice and viewport index, a ushort and a uint. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool amplified [[function_constant(0)]];

struct MeshAmplifiedVertex {
    float4 position [[position]];
    float2 uv;
};

struct MeshAmplifiedPrimitive {
    uint view [[flat]];
};

using MeshAmplifiedMesh =
    metal::mesh<MeshAmplifiedVertex, MeshAmplifiedPrimitive, 3, 1, topology::triangle>;

[[mesh]] void mesh_amplified(MeshAmplifiedMesh out,
                             ushort amp [[amplification_id]],
                             ushort count [[amplification_count]],
                             uint tid [[thread_index_in_threadgroup]]) {
    if (tid < 3) {
        MeshAmplifiedVertex v;
        v.position = float4(float(tid & 1u) + float(amp), float(tid >> 1), 0.5, 1.0);
        v.uv = float2(float(count));
        out.set_vertex(tid, v);
        out.set_index(tid, uchar(tid));
    }
    if (tid == 0) {
        MeshAmplifiedPrimitive p;
        p.view = uint(amp) * 16u + uint(count);
        out.set_primitive(0, p);
        out.set_primitive_count(1);
    }
}

[[mesh]] void mesh_amplified_gated(MeshAmplifiedMesh out,
                                   ushort amp [[amplification_id, function_constant(amplified)]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    if (tid < 3) {
        MeshAmplifiedVertex v;
        v.position = float4(float(tid & 1u), float(tid >> 1), 0.5, 1.0);
        v.uv = float2(0.0);
        out.set_vertex(tid, v);
        out.set_index(tid, uchar(tid));
    }
    if (tid == 0) {
        MeshAmplifiedPrimitive p;
        p.view = amplified ? uint(amp) : 9u;
        out.set_primitive(0, p);
        out.set_primitive_count(1);
    }
}

struct MeshLayeredPrimitive {
    uint view [[flat]];
    ushort layer [[render_target_array_index]];
    uint viewport [[viewport_array_index]];
};

using MeshLayeredMesh =
    metal::mesh<MeshAmplifiedVertex, MeshLayeredPrimitive, 3, 1, topology::triangle>;

[[mesh]] void mesh_layered(MeshLayeredMesh out,
                           constant uint2 &target [[buffer(0)]],
                           uint tid [[thread_index_in_threadgroup]]) {
    if (tid < 3) {
        MeshAmplifiedVertex v;
        v.position = float4(float(tid & 1u), float(tid >> 1), 0.5, 1.0);
        v.uv = float2(float(tid));
        out.set_vertex(tid, v);
        out.set_index(tid, uchar(tid));
    }
    if (tid == 0) {
        MeshLayeredPrimitive p;
        p.view = target.x ^ target.y;
        p.layer = ushort(target.x);
        p.viewport = target.y;
        out.set_primitive(0, p);
        out.set_primitive_count(1);
    }
}
