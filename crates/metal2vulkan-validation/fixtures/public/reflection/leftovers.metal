// Owned fixture: the bindings whose reflection no
// earlier probe measured: an explicit imageblock argument of a tile function and of a compute
// kernel, a fragment function's imageblock data (read only: Metal rejects raster-order-group
// members in a fragment function's output), render pipeline states (one and an array of two)
// inside an argument buffer, implicit indices after arguments a function constant gates, and
// indices a function constant computes. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ReflectionCell {
    float4 colour;
    half weight;
};

// An explicit imageblock in a tile function.
kernel void reflection_tile_explicit(imageblock<ReflectionCell, imageblock_layout_explicit> ib,
                                     constant float &k [[buffer(0)]],
                                     ushort2 lid [[thread_position_in_threadgroup]]) {
    threadgroup_imageblock ReflectionCell *cell = ib.data(lid);
    cell->colour = float4(float(lid.x), float(lid.y), k, 1.0f);
    cell->weight = half(lid.x);
}

// An explicit imageblock in a compute kernel, read back to a buffer.
kernel void reflection_kernel_explicit(imageblock<ReflectionCell> ib,
                                       device float *out [[buffer(0)]],
                                       ushort2 lid [[thread_position_in_threadgroup]],
                                       uint2 gid [[thread_position_in_grid]]) {
    threadgroup_imageblock ReflectionCell *cell = ib.data(lid);
    cell->colour = float4(float(gid.x));
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    out[gid.x] = ib.data(lid ^ ushort2(1, 0))->colour.x;
}

struct ReflectionPixel {
    half4 c [[raster_order_group(0)]];
    ushort k [[raster_order_group(0)]];
};

struct ReflectionPoint {
    float4 position [[position]];
};

vertex ReflectionPoint reflection_cover(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return ReflectionPoint { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

// A fragment function that reads its imageblock data and writes a colour.
fragment half4 reflection_fragment_data(ReflectionPixel in [[imageblock_data]]) {
    return in.c + half4(half(in.k));
}

struct ReflectionPipelines {
    render_pipeline_state render [[id(0)]];
    compute_pipeline_state compute [[id(1)]];
    uint scale [[id(2)]];
    array<render_pipeline_state, 2> renders [[id(3)]];
};

// Render pipeline states (one, and an array of two) inside an argument buffer.
kernel void reflection_pipeline_states(device const ReflectionPipelines &p [[buffer(0)]],
                                       device uint *out [[buffer(1)]],
                                       uint i [[thread_position_in_grid]]) {
    out[i] = i * p.scale;
}

constant bool reflection_gate [[function_constant(0)]];

// Implicit indices after arguments a constant gates: Apple's compiler numbers them in
// declaration order whatever the constant, as literals.
kernel void reflection_gated_indices(device float *gated [[function_constant(reflection_gate)]],
                                     device float *after,
                                     texture2d<float, access::read> gated_texture [[function_constant(reflection_gate)]],
                                     texture2d<float, access::read> texture,
                                     uint i [[thread_position_in_grid]]) {
    float v = texture.read(uint2(i, 0)).x;
    if (reflection_gate) {
        v += gated[i] + gated_texture.read(uint2(i, 0)).x;
    }
    after[i] = v;
}

constant int reflection_base [[function_constant(1)]];

// Indices a function constant computes: the module states each as a global its static
// initializer computes from the constant, which a library answers before any value is given.
kernel void reflection_computed_indices(device const float *first [[buffer(reflection_base)]],
                                        device float *second [[buffer(reflection_base + 2)]],
                                        texture2d<float, access::read> image [[texture(reflection_base * 2)]],
                                        uint i [[thread_position_in_grid]]) {
    second[i] = first[i] + image.read(uint2(i, 0)).x;
}
