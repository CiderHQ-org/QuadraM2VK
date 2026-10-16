// Owned MSL: texture and sampler handles as values.
// Every function is loop-free.
#include <metal_stdlib>
using namespace metal;

// A texture selected between two parameters: a read, queries, and is_null through the handle.
kernel void select_read(texture2d<float> a [[texture(0)]],
                        texture2d<float> b [[texture(1)]],
                        device float4 *out [[buffer(0)]],
                        constant uint &pick [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    texture2d<float> t = (gid & 1) == pick ? a : b;
    out[gid] = t.read(uint2(gid, 0))
        + float4(t.get_width(), t.get_height(), is_null_texture(t) ? 1.0 : 0.0, 0.0);
}

// A texture merged at a join: the phi of two handles.
kernel void phi_read(texture2d<float> a [[texture(0)]],
                     texture2d<float> b [[texture(1)]],
                     device float4 *out [[buffer(0)]],
                     constant uint &pick [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    texture2d<float> t = b;
    if (gid < pick) {
        out[gid + 64] = float4(1.0);
        t = a;
    }
    out[gid] = t.read(uint2(gid, 0));
}

// A storage texture or the null texture: a write through null is dropped.
kernel void select_null_write(texture2d<float, access::write> t [[texture(0)]],
                              uint gid [[thread_position_in_grid]]) {
    texture2d<float, access::write> w = (gid & 1) ? t : texture2d<float, access::write>();
    w.write(float4(1.0), uint2(gid, 0));
}

// An array of textures, indexed dynamically and at a constant.
kernel void array_read(array<texture2d<float>, 4> ts [[texture(0)]],
                       device float4 *out [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    out[gid] = ts[gid & 3].read(uint2(gid, 0)) + float4(ts[2].get_width());
}

// A sampler array indexed dynamically, and two selected samplers sampling two selected
// textures, at explicit levels (not tangled: a switch at the use).
kernel void sampler_select(texture2d<float> t [[texture(0)]],
                           texture2d<float> u [[texture(1)]],
                           array<sampler, 2> ss [[sampler(0)]],
                           sampler s0 [[sampler(2)]],
                           sampler s1 [[sampler(3)]],
                           device float4 *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    float2 uv = float2(gid) * 0.25;
    sampler s = (gid & 2) ? s0 : s1;
    texture2d<float> v = (gid & 4) ? t : u;
    out[gid] = t.sample(ss[gid & 1], uv, level(0.0)) + v.sample(s, uv, level(1.0));
}

// Implicit-LOD samples through a selected texture (tangled): the gradients are hoisted
// before the switch, the bias scaling them.
fragment float4 fragment_select_sample(float4 pos [[position]],
                                       texture2d<float> a [[texture(0)]],
                                       texture2d<float> b [[texture(1)]],
                                       sampler s [[sampler(0)]]) {
    texture2d<float> t = pos.x > 8.0 ? a : b;
    float2 uv = pos.xy * 0.0625;
    return t.sample(s, uv) + t.sample(s, uv, bias(0.5));
}

// An LOD query through a selected texture: no gradient form, so each candidate's query runs
// before the switch, at the call's position, and the switch selects one.
fragment float4 fragment_select_lod(float4 pos [[position]],
                                    texture2d<float> a [[texture(0)]],
                                    texture2d<float> b [[texture(1)]],
                                    sampler s [[sampler(0)]]) {
    texture2d<float> t = pos.x > 8.0 ? a : b;
    return float4(t.calculate_clamped_lod(s, pos.xy * 0.0625));
}

struct Resources {
    texture2d<float> t;
};

// A texture loaded from an argument buffer: the heap of argument-buffer textures (not implemented).
kernel void argument_buffer_read(constant Resources &res [[buffer(0)]],
                                 device float4 *out [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
    out[gid] = res.t.read(uint2(gid, 0));
}

// Argument-buffer members reached through one dynamic element index: an element
// of a texture-array member, written through the storage heap, and the texture of an element of
// an array of argument buffers, tested for null and read through the sampled heap.
struct Slices {
    array<texture2d<float, access::write>, 8> slices;
};

kernel void argument_buffer_array_write(constant Slices &res [[buffer(0)]],
                                        uint gid [[thread_position_in_grid]]) {
    res.slices[gid & 7].write(float4(1.0), uint2(gid, 0));
}

struct Atlas {
    texture2d<float> t;
    uchar4 swizzle;
};

kernel void argument_buffer_struct_array(constant Atlas *atlas [[buffer(0)]],
                                         device float4 *out [[buffer(1)]],
                                         uint gid [[thread_position_in_grid]]) {
    out[gid] = is_null_texture(atlas[gid].t) ? float4(atlas[gid].swizzle)
                                             : atlas[gid].t.read(uint2(gid, 0));
}

// A depth texture read through an argument buffer, written to a storage texture of the same
// argument buffer: the depth is the float heap class's first component.
struct DepthBlit {
    depth2d<float, access::read> source;
    texture2d<float, access::write> target;
};

kernel void argument_buffer_depth_read(constant DepthBlit &blit [[buffer(0)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    blit.target.write(float4(blit.source.read(gid)), gid);
}

// Argument-buffer textures sampled through constexpr samplers: each heap element
// through a separate heap sampler. A kernel copies through a normalized linear sampler and a
// nearest pixel sampler (an exact fetch), and an integer texture through a nearest sampler.
constexpr sampler linear_clamp(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler nearest_clamp(coord::normalized, filter::nearest, address::clamp_to_edge);
constexpr sampler pixel_nearest(coord::pixel, filter::nearest, address::clamp_to_edge);

struct Copy {
    texture2d<float> source;
    texture2d<float, access::write> target;
};

kernel void argument_buffer_sample(constant Copy &copy [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    float2 size = float2(copy.target.get_width(), copy.target.get_height());
    float2 uv = (float2(gid) + 0.5) / size;
    copy.target.write(copy.source.sample(linear_clamp, uv)
                          + copy.source.sample(pixel_nearest, float2(gid) + 0.5),
                      gid);
}

struct IntCopy {
    texture2d<int> source;
    texture2d<int, access::write> target;
};

kernel void argument_buffer_sample_int(constant IntCopy &copy [[buffer(0)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    float2 uv = (float2(gid) + 0.5) / float2(copy.target.get_width(), copy.target.get_height());
    copy.target.write(copy.source.sample(nearest_clamp, uv), gid);
}

// A fragment's implicit-LOD samples of a cube, a 2D texture, and a depth texture of one
// argument buffer, and of an element of a texture-array member picked dynamically.
struct Environment {
    texturecube<half> sky;
    texture2d<half> albedo;
    depth2d<float> shadow;
    array<texture2d<float>, 4> layers;
};

fragment float4 argument_buffer_fragment_sample(float4 pos [[position]],
                                                constant Environment &env [[buffer(0)]],
                                                constant uint &pick [[buffer(1)]]) {
    float2 uv = pos.xy * 0.0625;
    float3 dir = float3(uv - 1.0, 1.0);
    half4 lit = env.sky.sample(linear_clamp, dir) * env.albedo.sample(linear_clamp, uv);
    return float4(lit) * env.shadow.sample(linear_clamp, uv)
        + env.layers[pick & 3].sample(linear_clamp, uv, bias(0.5));
}

// A heap texture sampled through a runtime sampler: a heap sampler binding the binder writes with
// the sampler bound at the slot.
kernel void argument_buffer_runtime_sampler(constant Resources &res [[buffer(0)]],
                                            sampler s [[sampler(0)]],
                                            device float4 *out [[buffer(1)]],
                                            uint gid [[thread_position_in_grid]]) {
    out[gid] = res.t.sample(s, float2(gid) * 0.25, level(0.0));
}

// A heap depth texture sampled with comparison: a member of the depth-comparison heap class
//.
constexpr sampler shadow_less(coord::normalized, filter::linear, compare_func::less);

struct Shadow {
    depth2d<float> map;
};

kernel void argument_buffer_sample_compare(constant Shadow &shadow [[buffer(0)]],
                                           device float *out [[buffer(1)]],
                                           uint gid [[thread_position_in_grid]]) {
    out[gid] = shadow.map.sample_compare(shadow_less, float2(gid) * 0.25, 0.5, level(0));
}

// A heap texture sampled with an opaque-black border: every member the sampled array holds must
// read that border exactly through its view.
constexpr sampler black_border(coord::normalized, filter::linear, address::clamp_to_border,
                               border_color::opaque_black);

kernel void argument_buffer_sample_border(constant Resources &res [[buffer(0)]],
                                          device float4 *out [[buffer(1)]],
                                          uint gid [[thread_position_in_grid]]) {
    out[gid] = res.t.sample(black_border, float2(gid) * 0.25 - 1.0);
}
