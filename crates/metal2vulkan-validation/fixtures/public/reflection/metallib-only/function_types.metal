// Owned fixture: the function kinds whose metallib `TYPE` tag
// and MTLFunction.functionType the other reflection fixtures do not show (mesh, object and
// intersection functions), and a post-tessellation vertex function whose patch states no
// control-point count. Written for this repository; no third-party source. Loop-free.
//
// Only its metallib's answers are compared (`metallib-tags.txt`, `metal-answers.txt` one level
// up), so it lives outside the directory scripts/regen-air-fixtures compiles and has no committed
// `.ll` companions: its AIR is not AIR the product decodes. Apple's compiler (Metal 32023.921)
// passes the mesh function's `mesh<...>` handle as `ptr addrspace(7)` and the object function's
// `object_data` payload as `ptr addrspace(6)`, and lists the entries under `!air.mesh`,
// `!air.object` and `!air.intersection`, none of which the decoder accepts (mesh and object
// functions are not decoded yet; intersection functions are ray tracing, out of scope), and every
// committed `.ll` must decode (validation/tests/fixtures_assemble_with_the_pinned_llvm.rs).
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct TypesVertex {
    float4 position [[position]];
};

using TypesMesh = mesh<TypesVertex, void, 3, 1, topology::triangle>;

[[mesh]]
void types_mesh(TypesMesh m, uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        m.set_primitive_count(1);
    }
    if (tid < 3) {
        TypesVertex v;
        v.position = float4(float(tid), 0.0, 0.0, 1.0);
        m.set_vertex(tid, v);
        m.set_index(tid, uchar(tid));
    }
}

struct TypesPayload {
    uint n;
};

[[object]]
void types_object(object_data TypesPayload &payload [[payload]],
                  mesh_grid_properties grid,
                  uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        payload.n = 1;
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

struct TypesHit {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

[[intersection(bounding_box)]]
TypesHit types_intersection(float min_distance [[min_distance]]) {
    TypesHit h;
    h.accept = true;
    h.distance = min_distance;
    return h;
}

struct TypesCorner {
    float4 p [[attribute(0)]];
};

struct TypesCorners {
    patch_control_point<TypesCorner> cps;
};

[[patch(quad)]]
vertex float4 types_quad_any(TypesCorners in [[stage_in]], float2 uv [[position_in_patch]]) {
    return in.cps[0].p * uv.x;
}
