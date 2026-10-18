// Owned fixture: functions that use, or only declare, a direct texture or
// sampler argument, for the runtime lane's states that support indirect command buffers. Metal
// answers nil with an NSError for such a state whose functions use one,
// and makes it where they only declare one. Written from scratch for the runtime lane;
// `scripts/regen-air-fixtures/regen-air-fixtures.sh --dir examples icb_direct_arguments` compiles
// it. No translator sweep reads this directory.
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float4 color [[flat]];
};

constant float2 corners[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };

// A triangle over the target, reading nothing.
vertex Varyings plain_vertex(uint id [[vertex_id]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = float4(0.25, 0.5, 0.75, 1.0);
    return out;
}

// The triangle, its color read from a texture.
vertex Varyings texture_vertex(texture2d<float> tint [[texture(0)]], uint id [[vertex_id]]) {
    Varyings out;
    out.position = float4(corners[id], 0.0, 1.0);
    out.color = tint.read(uint2(0, 0));
    return out;
}

// The vertex color.
fragment float4 plain_fragment(Varyings in [[stage_in]]) {
    return in.color;
}

// A texel of a texture.
fragment float4 texture_fragment(Varyings in [[stage_in]], texture2d<float> source [[texture(0)]]) {
    return source.read(uint2(0, 0));
}

// A sample of a texture through a sampler argument.
fragment float4 sampler_fragment(Varyings in [[stage_in]],
                                 texture2d<float> source [[texture(0)]],
                                 sampler linear [[sampler(0)]]) {
    return source.sample(linear, float2(0.5, 0.5));
}

// The vertex color, beside a texture and a sampler argument it never reads.
fragment float4 unused_arguments_fragment(Varyings in [[stage_in]],
                                          texture2d<float> source [[texture(0)]],
                                          sampler linear [[sampler(0)]]) {
    return in.color;
}

// One triangle over the target, from thread 0 of the mesh threadgroup.
using Triangle = metal::mesh<Varyings, void, 3, 1, topology::triangle>;

static void triangle(Triangle out, float4 color) {
    out.set_primitive_count(1);
    for (uint i = 0; i < 3; i++) {
        Varyings v;
        v.position = float4(corners[i], 0.0, 1.0);
        v.color = color;
        out.set_vertex(i, v);
        out.set_index(i, i);
    }
}

[[mesh]] void plain_mesh(Triangle out, uint index [[thread_index_in_threadgroup]]) {
    if (index == 0) {
        triangle(out, float4(0.25, 0.5, 0.75, 1.0));
    }
}

[[mesh]] void texture_mesh(Triangle out, uint index [[thread_index_in_threadgroup]],
                           texture2d<float> tint [[texture(0)]]) {
    if (index == 0) {
        triangle(out, tint.read(uint2(0, 0)));
    }
}

struct Payload {
    float4 color;
};

// One mesh threadgroup.
[[object]] void plain_object(object_data Payload &payload [[payload]], mesh_grid_properties grid,
                             uint index [[thread_index_in_threadgroup]]) {
    if (index == 0) {
        payload.color = float4(1.0);
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

// One mesh threadgroup, its payload read from a texture.
[[object]] void texture_object(object_data Payload &payload [[payload]], mesh_grid_properties grid,
                               uint index [[thread_index_in_threadgroup]],
                               texture2d<float> tint [[texture(0)]]) {
    if (index == 0) {
        payload.color = tint.read(uint2(0, 0));
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

// A texel of a texture, written to a buffer.
kernel void texture_kernel(texture2d<float, access::read> source [[texture(0)]],
                           device float4 *out [[buffer(0)]]) {
    out[0] = source.read(uint2(0, 0));
}

// A buffer's word, written to another.
kernel void buffer_kernel(device const float4 *source [[buffer(1)]], device float4 *out [[buffer(0)]]) {
    out[0] = source[0];
}
