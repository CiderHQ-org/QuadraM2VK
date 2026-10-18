// Owned fixture: vertex, mesh and object functions that use a direct
// texture or sampler argument, by where the value read goes once Metal links the pipeline, and the
// fragment functions that read their varyings or drop them. Metal refuses a state that supports
// indirect command buffers whose vertex or mesh function's texel reaches the position or another
// built-in output, a varying the fragment function reads, a store, or a branch, switch or loop
// trip count those depend on, and makes one whose texel reaches only a varying no fragment input
// reads (whose probes compile these bodies from source). Written from scratch for the runtime lane;
// `scripts/regen-air-fixtures/regen-air-fixtures.sh --dir examples icb_linked_arguments` compiles
// it. No translator sweep reads this directory.
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float4 color;
};

struct PositionOnly {
    float4 position [[position]];
};

struct TwoVaryings {
    float4 position [[position]];
    float4 color;
    float4 other;
};

struct PointVaryings {
    float4 position [[position]];
    float size [[point_size]];
    float4 color;
};

constant float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };

// The texel is the varying `color`.
vertex Varyings varying_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = tint.read(uint2(1, 0));
    return out;
}

// The texel, sampled through a sampler argument, is the varying `color`.
vertex Varyings sampled_varying_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                                       sampler smp [[sampler(0)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = tint.sample(smp, float2(0.75, 0.5), level(0));
    return out;
}

// The texel is the position's depth only.
vertex Varyings depth_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]]) {
    Varyings out;
    out.position = float4(corners[id], tint.read(uint2(1, 0)).x, 1.0);
    out.color = 0.0;
    return out;
}

// The texel is the depth of a function that returns its position alone.
vertex PositionOnly position_only_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]]) {
    PositionOnly out;
    out.position = float4(corners[id], tint.read(uint2(1, 0)).x, 1.0);
    return out;
}

// The texel is stored to a buffer; the outputs are constants.
vertex Varyings store_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                             device float4* out_buffer [[buffer(0)]]) {
    out_buffer[id] = tint.read(uint2(1, 0));
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = 0.0;
    return out;
}

// The texel is stored to a buffer by a function that returns nothing.
vertex void void_store_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                              device float4* out_buffer [[buffer(0)]]) {
    out_buffer[id] = tint.read(uint2(1, 0));
}

// The sample's result is discarded.
vertex Varyings discarded_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                                 sampler smp [[sampler(0)]]) {
    float4 unused = tint.sample(smp, float2(0.75, 0.5), level(0));
    (void)unused;
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = 0.0;
    return out;
}

// A triangle over the target, reading nothing.
vertex Varyings plain_vertex(uint id [[vertex_id]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = 0.0;
    return out;
}

// A function that returns its position alone, not in a struct.
vertex float4 position_value_vertex(uint id [[vertex_id]]) {
    return float4(corners[id], 0.0, 1.0);
}

// The texel decides a branch whose arm loads the varying and stores to the buffer.
vertex Varyings branch_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                              device float4* data [[buffer(1)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    if (tint.read(uint2(1, 0)).x > 0.5) {
        out.color = data[id];
        data[id + 8] = 1.0;
    } else {
        out.color = float4(0.25);
    }
    return out;
}

// The texel decides whether a store happens.
vertex Varyings guarded_store_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                                     device float4* data [[buffer(1)]]) {
    if (tint.read(uint2(1, 0)).x > 0.5) {
        data[id] = float4(1.0);
    }
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = 0.0;
    return out;
}

// The texel is a loop's trip count; the loop's result is the varying.
vertex Varyings trip_count_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                                  constant float4* weights [[buffer(1)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    float4 sum = 0.0;
    uint count = uint(tint.read(uint2(1, 0)).x * 8.0);
    for (uint i = 0; i < count; i++) {
        sum = sum * weights[i & 3] + weights[(i + 1) & 3];
    }
    out.color = sum;
    return out;
}

// The texel chooses the varying's value by the ternary operator (a select: data flow).
vertex Varyings select_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = tint.read(uint2(1, 0)).x > 0.5 ? float4(1.0) : float4(0.0);
    return out;
}

// The texel is the point size, a built-in output.
vertex PointVaryings point_size_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]]) {
    PointVaryings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.size = tint.read(uint2(1, 0)).x;
    out.color = 0.0;
    return out;
}

// The texel is the varying `other`; `color` is a constant.
vertex TwoVaryings other_varying_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]]) {
    TwoVaryings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = 0.5;
    out.other = tint.read(uint2(1, 0));
    return out;
}

// The texel decides a branch whose arms (too costly to speculate) compute the varying.
vertex Varyings dead_branch_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                                   device float4* data [[buffer(1)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    if (tint.read(uint2(1, 0)).x > 0.5) {
        out.color = data[id] * data[id + 1] + data[id + 2] * data[id + 3] - data[id + 4];
    } else {
        out.color = data[id + 5] / data[id + 6] - data[id + 7] / data[id + 8] + data[id + 9];
    }
    return out;
}

// The texel's value in one arm of a branch on the vertex id, merged into the varying.
vertex Varyings merged_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                              device float4* data [[buffer(1)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    if (id > 1) {
        out.color = tint.read(uint2(1, 0)) * data[id] + data[id + 1] * data[id + 2] - data[id + 3];
    } else {
        out.color = data[id + 5] / data[id + 6] - data[id + 7] / data[id + 8] + data[id + 9];
    }
    return out;
}

// The texel selects a switch's case, each case a different store.
vertex Varyings switch_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                              device float4* data [[buffer(1)]]) {
    switch (uint(tint.read(uint2(1, 0)).x * 4.0)) {
        case 0: data[id] = float4(1.0); break;
        case 1: data[id + 8] = float4(2.0); break;
        case 2: data[id + 16] = float4(3.0); break;
        default: break;
    }
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = 0.0;
    return out;
}

// A loop whose trip count comes from a buffer; its body folds the texel into the varying.
vertex Varyings loop_body_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                                 constant uint* count [[buffer(1)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    float4 sum = 0.0;
    for (uint i = 0; i < count[0]; i++) {
        sum = sum * 0.5 + tint.read(uint2(i & 1, 0));
    }
    out.color = sum;
    return out;
}

// The same loop, storing each count too.
vertex Varyings storing_loop_vertex(uint id [[vertex_id]], texture2d<float> tint [[texture(0)]],
                                    constant uint* count [[buffer(1)]],
                                    device float4* data [[buffer(2)]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    float4 sum = 0.0;
    for (uint i = 0; i < count[0]; i++) {
        sum = sum * 0.5 + tint.read(uint2(i & 1, 0));
        data[i] = float4(float(i));
    }
    out.color = sum;
    return out;
}

// Reads the varying `color`.
fragment float4 varying_fragment(Varyings in [[stage_in]]) {
    return in.color;
}

// Declares the varyings and reads none.
fragment float4 declaring_fragment(Varyings in [[stage_in]]) {
    return float4(0.25, 0.5, 0.75, 1.0);
}

// Declares no stage input.
fragment float4 inputless_fragment() {
    return float4(0.25, 0.5, 0.75, 1.0);
}

// Reads the position's depth.
fragment float4 depth_fragment(PositionOnly in [[stage_in]]) {
    return float4(in.position.z, 0.0, 0.0, 1.0);
}

// Reads `color` of two varyings.
fragment float4 first_varying_fragment(TwoVaryings in [[stage_in]]) {
    return in.color;
}

// Declares a point's varyings and reads none.
fragment float4 point_fragment(PointVaryings in [[stage_in]]) {
    return float4(0.25, 0.5, 0.75, 1.0);
}

// Fragment functions that use a direct texture: Metal refuses a state that supports indirect
// command buffers where the texel reaches a side effect, a depth, stencil or sample mask output, or
// a color channel its attachment writes, and makes one where it reaches only a color the pipeline
// has no attachment for, a channel the write mask leaves out, or a dual-source second color without
// blending. The texel is the color.
fragment float4 texel_fragment(Varyings in [[stage_in]], texture2d<float> tint [[texture(0)]]) {
    return tint.read(uint2(1, 0));
}

// The texel is the color's alpha alone.
fragment float4 alpha_fragment(Varyings in [[stage_in]], texture2d<float> tint [[texture(0)]]) {
    return float4(0.25, 0.5, 0.75, tint.read(uint2(1, 0)).x);
}

// The texel is the color's red alone.
fragment float4 red_fragment(Varyings in [[stage_in]], texture2d<float> tint [[texture(0)]]) {
    return float4(tint.read(uint2(1, 0)).x, 0.5, 0.75, 1.0);
}

// The texel decides a discard; the color is constant.
fragment float4 discarding_fragment(Varyings in [[stage_in]],
                                    texture2d<float> tint [[texture(0)]]) {
    if (tint.read(uint2(1, 0)).x > 0.5) {
        discard_fragment();
    }
    return float4(0.25, 0.5, 0.75, 1.0);
}

// The texel goes to a device store; the color is constant.
fragment float4 storing_fragment(Varyings in [[stage_in]], texture2d<float> tint [[texture(0)]],
                                 device float4* out_buffer [[buffer(0)]]) {
    out_buffer[uint(in.position.x)] = tint.read(uint2(1, 0));
    return float4(0.25, 0.5, 0.75, 1.0);
}

struct TwoColors {
    float4 first [[color(0)]];
    float4 second [[color(1)]];
};

// The texel is the color of attachment 1; attachment 0's is constant.
fragment TwoColors second_color_fragment(Varyings in [[stage_in]],
                                         texture2d<float> tint [[texture(0)]]) {
    TwoColors out;
    out.first = float4(0.5);
    out.second = tint.read(uint2(1, 0));
    return out;
}

struct DualSource {
    float4 first [[color(0), index(0)]];
    float4 second [[color(0), index(1)]];
};

// The texel is a dual-source blend's second color alone.
fragment DualSource dual_source_fragment(Varyings in [[stage_in]],
                                         texture2d<float> tint [[texture(0)]]) {
    DualSource out;
    out.first = float4(0.5);
    out.second = tint.read(uint2(1, 0));
    return out;
}

struct DepthOutput {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

// The texel is the depth output alone.
fragment DepthOutput depth_writing_fragment(Varyings in [[stage_in]],
                                            texture2d<float> tint [[texture(0)]]) {
    DepthOutput out;
    out.color = float4(0.5);
    out.depth = tint.read(uint2(1, 0)).x;
    return out;
}

struct MaskOutput {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

// The texel is the sample mask alone.
fragment MaskOutput mask_fragment(Varyings in [[stage_in]], texture2d<float> tint [[texture(0)]]) {
    MaskOutput out;
    out.color = float4(0.5);
    out.mask = uint(tint.read(uint2(1, 0)).x * 15.0);
    return out;
}

struct StencilOutput {
    float4 color [[color(0)]];
    uint stencil [[stencil]];
};

// The texel is the stencil reference alone.
fragment StencilOutput stencil_fragment(Varyings in [[stage_in]],
                                        texture2d<float> tint [[texture(0)]]) {
    StencilOutput out;
    out.color = float4(0.5);
    out.stencil = uint(tint.read(uint2(1, 0)).x * 255.0);
    return out;
}

// The varying decides a branch whose arms (too costly to speculate) compute the color: with a
// color attachment its pipeline keeps the read through the branch.
fragment float4 branching_fragment(Varyings in [[stage_in]],
                                   device const float4* weights [[buffer(0)]]) {
    if (in.color.x > 0.5) {
        return weights[0] * weights[1] + weights[2] * weights[3] - weights[4];
    }
    return weights[5] / weights[6] - weights[7] / weights[8] + weights[9];
}

// The varying is the color alone, beside a second color from a constant: with only attachment 1,
// the read is dropped.
fragment TwoColors varying_first_color_fragment(Varyings in [[stage_in]]) {
    TwoColors out;
    out.first = in.color;
    out.second = float4(0.5);
    return out;
}

// Returns nothing and uses no texture.
vertex void void_plain_vertex(uint id [[vertex_id]], device float4* out_buffer [[buffer(0)]]) {
    out_buffer[id] = float4(1.0);
}

// Mesh and object functions.
struct MeshVertex {
    float4 position [[position]];
    float4 color;
};

using TriangleMesh = metal::mesh<MeshVertex, void, 3, 1, topology::triangle>;

static void triangle(TriangleMesh output, float4 color, float depth) {
    output.set_primitive_count(1);
    for (uint i = 0; i < 3; i++) {
        MeshVertex v;
        v.position = float4(corners[i], depth, 1.0);
        v.color = color;
        output.set_vertex(i, v);
        output.set_index(i, i);
    }
}

// The texel is the mesh's varying `color`.
[[mesh]] void varying_mesh(TriangleMesh output, uint index [[thread_index_in_threadgroup]],
                           texture2d<float> tint [[texture(0)]]) {
    if (index == 0) {
        triangle(output, tint.read(uint2(1, 0)), 0.0);
    }
}

// The texel is the mesh's depth.
[[mesh]] void depth_mesh(TriangleMesh output, uint index [[thread_index_in_threadgroup]],
                         texture2d<float> tint [[texture(0)]]) {
    if (index == 0) {
        triangle(output, float4(0.0), tint.read(uint2(1, 0)).x);
    }
}

// A mesh reading nothing.
[[mesh]] void plain_mesh(TriangleMesh output, uint index [[thread_index_in_threadgroup]]) {
    if (index == 0) {
        triangle(output, float4(0.0), 0.0);
    }
}

struct Payload {
    float4 color;
};

// The texel is the payload.
[[object]] void texture_object(object_data Payload& payload [[payload]], mesh_grid_properties grid,
                               uint index [[thread_index_in_threadgroup]],
                               texture2d<float> tint [[texture(0)]]) {
    if (index == 0) {
        payload.color = tint.read(uint2(1, 0));
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

// The payload is a constant.
[[object]] void plain_object(object_data Payload& payload [[payload]], mesh_grid_properties grid,
                             uint index [[thread_index_in_threadgroup]]) {
    if (index == 0) {
        payload.color = float4(0.5);
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

// The payload is the mesh's varying `color`.
[[mesh]] void payload_mesh(TriangleMesh output, uint index [[thread_index_in_threadgroup]],
                           const object_data Payload& payload [[payload]]) {
    if (index == 0) {
        triangle(output, payload.color, 0.0);
    }
}

// Writes its vertices' positions alone: a fragment function's `color` is unwritten.
struct PositionMeshVertex {
    float4 position [[position]];
};

using PositionMesh = metal::mesh<PositionMeshVertex, void, 3, 1, topology::triangle>;

[[mesh]] void position_mesh(PositionMesh output, uint index [[thread_index_in_threadgroup]]) {
    if (index == 0) {
        output.set_primitive_count(1);
        for (uint i = 0; i < 3; i++) {
            PositionMeshVertex v;
            v.position = float4(corners[i], 0.0, 1.0);
            output.set_vertex(i, v);
            output.set_index(i, i);
        }
    }
}

// A post-tessellation vertex function writing its position alone: a fragment function's `color`
// is unwritten.
[[patch(triangle, 3)]] vertex float4 post_tess_vertex(float3 bary [[position_in_patch]]) {
    return float4(bary, 1.0);
}

// Reads the mesh's varying `color`.
fragment float4 mesh_varying_fragment(MeshVertex in [[stage_in]]) {
    return in.color;
}

// Declares the mesh's varyings and reads none.
fragment float4 mesh_declaring_fragment(MeshVertex in [[stage_in]]) {
    return float4(0.25, 0.5, 0.75, 1.0);
}
