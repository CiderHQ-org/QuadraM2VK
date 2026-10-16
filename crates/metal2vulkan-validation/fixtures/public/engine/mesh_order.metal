// Owned fixture: the draws whose answers Metal probes 1 to 3
// measured on an M4 and an M3: primitive order (object threadgroup, then mesh threadgroup, then
// slot), set_primitive_count past the slots written, slots whose indices are not all written,
// primitive_culled, per-primitive data, the payload, and the fragment's primitive_id. The fragment
// function appends each primitive's id and primitive_id to its render targets (framebuffer fetch),
// so a pixel records the primitives that covered it, in order, and writes the last one's
// interpolated vertex data and per-primitive weight to a third. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct MeshOrderCtl {
    uint nprims;    // primitive slots written: 0 .. nprims - 1
    uint count;     // set_primitive_count
    uint cullmask;  // primitive_culled, per slot
    uint idxmask;   // slots whose three indices are written
    uint meshgrid;  // mesh threadgroups each object threadgroup asks for
    uint pad0;
    uint pad1;
    uint pad2;
};

struct MeshOrderPayload {
    uint base;
    uint scale;
};

struct MeshOrderVertex {
    float4 position [[position]];
    float shade;
};

struct MeshOrderPrimitive {
    uint id [[flat]];
    float weight;
    bool culled [[primitive_culled]];
};

using MeshOrderMesh = metal::mesh<MeshOrderVertex, MeshOrderPrimitive, 4, 4, topology::triangle>;

// Thread 0 of each object threadgroup writes the payload and asks for `meshgrid` mesh threadgroups.
[[object]] void mesh_order_object(object_data MeshOrderPayload& payload [[payload]],
                                  mesh_grid_properties grid,
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  constant MeshOrderCtl& ctl [[buffer(0)]]) {
    if (tid == 0) {
        payload.base = 16 * group.x;
        payload.scale = 1;
        grid.set_threadgroups_per_grid(uint3(ctl.meshgrid, 1, 1));
    }
}

// Four full-screen triangles; slot p's id is base + 4 * group + p + 1.
static void mesh_order_write(MeshOrderMesh out, uint tid, uint group, uint base,
                             constant MeshOrderCtl& ctl) {
    if (tid < 4) {
        MeshOrderVertex v;
        float2 a = float2(float(tid & 1u), float((tid >> 1) & 1u));
        v.position = tid == 3 ? float4(-1.0, -1.0, 0.5, 1.0) : float4(4.0 * a - 1.0, 0.5, 1.0);
        v.shade = float(tid);
        out.set_vertex(tid, v);
    }
    if (tid < ctl.nprims && tid < 4) {
        if (((ctl.idxmask >> tid) & 1u) != 0) {
            out.set_index(tid * 3 + 0, 0);
            out.set_index(tid * 3 + 1, 1);
            out.set_index(tid * 3 + 2, 2);
        }
        MeshOrderPrimitive p;
        p.id = base + 4 * group + tid + 1;
        p.weight = 0.25 * float(tid + 1);
        p.culled = ((ctl.cullmask >> tid) & 1u) != 0;
        out.set_primitive(tid, p);
    }
    if (tid == 0) {
        out.set_primitive_count(ctl.count);
    }
}

[[mesh]] void mesh_order_mesh(MeshOrderMesh out,
                              const object_data MeshOrderPayload& payload [[payload]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]],
                              constant MeshOrderCtl& ctl [[buffer(0)]]) {
    mesh_order_write(out, tid, group.x, payload.base * payload.scale, ctl);
}

// The same mesh without an object stage.
[[mesh]] void mesh_order_solo(MeshOrderMesh out,
                              uint tid [[thread_index_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]],
                              constant MeshOrderCtl& ctl [[buffer(0)]]) {
    mesh_order_write(out, tid, group.x, 0, ctl);
}

struct MeshOrderFragmentIn {
    MeshOrderVertex v;
    MeshOrderPrimitive p;
};

struct MeshOrderSeq {
    uint4 ids [[color(0)]];
    uint4 prims [[color(1)]];
    float4 data [[color(2)]];
};

// Appends `id` (mod 256) to the byte sequence `prev` holds: w counts, x..z hold four bytes each.
static uint4 mesh_order_append(uint4 prev, uint id) {
    uint n = prev.w;
    uint4 s = prev;
    id &= 255u;
    if (n < 4) {
        s.x = prev.x | (id << (8 * n));
    } else if (n < 8) {
        s.y = prev.y | (id << (8 * (n - 4)));
    } else if (n < 12) {
        s.z = prev.z | (id << (8 * (n - 8)));
    }
    s.w = n + 1;
    return s;
}

fragment MeshOrderSeq mesh_order_fragment(MeshOrderFragmentIn in [[stage_in]],
                                          uint prim [[primitive_id]],
                                          uint4 ids [[color(0)]], uint4 prims [[color(1)]]) {
    MeshOrderSeq o;
    o.ids = mesh_order_append(ids, in.p.id);
    o.prims = mesh_order_append(prims, prim + 1);
    o.data = float4(in.v.shade, in.p.weight, 0.0, 1.0);
    return o;
}
