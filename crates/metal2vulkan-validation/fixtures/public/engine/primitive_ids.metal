// Owned fixture: a fragment function that returns [[primitive_id]] beside the vertex and instance
// IDs of its primitive's flat (provoking) vertex, so a draw tells what Metal counts at each
// primitive of each topology, instance, start, base vertex, restart and indirect draw. Each vertex
// function places a vertex at a pixel position (buffer 0, or attribute 0 for `pi_attrib`), moved
// down a row stride per instance: `frame` is the target's width, height and row stride in pixels.
// `pi_points` also sizes its points, and `pi_counts` counts its invocations in a device atomic.
// `pi_five` returns the IDs to five color attachments, more than a device at Vulkan's floor draws
// at once.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct PiVertex {
    float4 position [[position]];
    uint vertex_id [[flat]];
    uint instance_id [[flat]];
};

struct PiPoint {
    float4 position [[position]];
    uint vertex_id [[flat]];
    uint instance_id [[flat]];
    float size [[point_size]];
};

static float4 place(float2 xy, uint iid, float4 frame) {
    float2 at = float2(xy.x, xy.y + float(iid) * frame.z);
    return float4(2.0 * at.x / frame.x - 1.0, 1.0 - 2.0 * at.y / frame.y, 0.0, 1.0);
}

vertex PiVertex pi_place(uint vid [[vertex_id]], uint iid [[instance_id]],
                         const device float2 *xy [[buffer(0)]],
                         constant float4 &frame [[buffer(1)]]) {
    return PiVertex { place(xy[vid], iid, frame), vid, iid };
}

vertex PiPoint pi_points(uint vid [[vertex_id]], uint iid [[instance_id]],
                         const device float2 *xy [[buffer(0)]],
                         constant float4 &frame [[buffer(1)]]) {
    return PiPoint { place(xy[vid], iid, frame), vid, iid, 1.0 };
}

struct PiIn {
    float2 xy [[attribute(0)]];
};

vertex PiVertex pi_attrib(PiIn in [[stage_in]], uint vid [[vertex_id]],
                          uint iid [[instance_id]], constant float4 &frame [[buffer(1)]]) {
    return PiVertex { place(in.xy, iid, frame), vid, iid };
}

vertex PiVertex pi_counts(uint vid [[vertex_id]], uint iid [[instance_id]],
                          const device float2 *xy [[buffer(0)]],
                          constant float4 &frame [[buffer(1)]],
                          device atomic_uint *count [[buffer(2)]]) {
    atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    return PiVertex { place(xy[vid], iid, frame), vid, iid };
}

fragment uint4 pi_ids(PiVertex in [[stage_in]], uint primitive [[primitive_id]]) {
    return uint4(primitive, in.vertex_id, in.instance_id, 1u);
}

struct PiFive {
    uint4 c0 [[color(0)]];
    uint4 c1 [[color(1)]];
    uint4 c2 [[color(2)]];
    uint4 c3 [[color(3)]];
    uint4 c4 [[color(4)]];
};

fragment PiFive pi_five(PiVertex in [[stage_in]], uint primitive [[primitive_id]]) {
    uint4 ids = uint4(primitive, in.vertex_id, in.instance_id, 1u);
    return PiFive { ids, ids + 1u, ids + 2u, ids + 3u, ids + 4u };
}
