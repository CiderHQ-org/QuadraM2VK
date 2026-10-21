// Owned fixture: draws of points, lines and triangles that a kernel encodes into an indirect
// command buffer and a render pass executes. Kernel `icb_merged_<type>` encodes command `tid`:
// vertex buffer 0 its own word of `marks`, vertex buffer 1 the frame, vertex buffer 2 the
// vertices' pixel positions, and the draw `draws[tid]` (vertex start, vertex count, instance
// count, base instance) of its type. Vertex `vid` of instance `iid` lies at `xy[vid]` moved down
// `iid` rows of `frame.z` pixels (`frame` is the target's width, height and row stride in
// pixels), sized one pixel, and carries its command's mark, `vid`, `iid` and the draw's base
// instance flat to the fragment function, which returns them. The vertex function writes no
// memory, and the fragment function reads no buffer.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct MergedCommands {
    command_buffer commands;
};

struct MergedDraw {
    uint start;
    uint count;
    uint instances;
    uint base;
};

struct MergedOut {
    float4 position [[position]];
    uint4 ids [[flat]];
    float size [[point_size]];
};

static void merged_encode(uint tid, device const MergedCommands &encoder,
                          device const MergedDraw *draws, constant uint *marks,
                          constant float4 *frame, device const float2 *xy, primitive_type type) {
    render_command command(encoder.commands, tid);
    command.set_vertex_buffer(marks + tid, 0);
    command.set_vertex_buffer(frame, 1);
    command.set_vertex_buffer(xy, 2);
    MergedDraw d = draws[tid];
    command.draw_primitives(type, d.start, d.count, d.instances, d.base);
}

kernel void icb_merged_points(device const MergedCommands &encoder [[buffer(0)]],
                              device const MergedDraw *draws [[buffer(1)]],
                              constant uint *marks [[buffer(2)]],
                              constant float4 *frame [[buffer(3)]],
                              device const float2 *xy [[buffer(4)]],
                              uint tid [[thread_position_in_grid]]) {
    merged_encode(tid, encoder, draws, marks, frame, xy, primitive_type::point);
}

kernel void icb_merged_lines(device const MergedCommands &encoder [[buffer(0)]],
                             device const MergedDraw *draws [[buffer(1)]],
                             constant uint *marks [[buffer(2)]],
                             constant float4 *frame [[buffer(3)]],
                             device const float2 *xy [[buffer(4)]],
                             uint tid [[thread_position_in_grid]]) {
    merged_encode(tid, encoder, draws, marks, frame, xy, primitive_type::line);
}

kernel void icb_merged_triangles(device const MergedCommands &encoder [[buffer(0)]],
                                 device const MergedDraw *draws [[buffer(1)]],
                                 constant uint *marks [[buffer(2)]],
                                 constant float4 *frame [[buffer(3)]],
                                 device const float2 *xy [[buffer(4)]],
                                 uint tid [[thread_position_in_grid]]) {
    merged_encode(tid, encoder, draws, marks, frame, xy, primitive_type::triangle);
}

vertex MergedOut icb_merged_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                   uint base [[base_instance]],
                                   constant uint &mark [[buffer(0)]],
                                   constant float4 &frame [[buffer(1)]],
                                   device const float2 *xy [[buffer(2)]]) {
    float2 at = xy[vid] + float2(0.0, float(iid) * frame.z);
    MergedOut out;
    out.position = float4(2.0 * at.x / frame.x - 1.0, 1.0 - 2.0 * at.y / frame.y, 0.0, 1.0);
    out.ids = uint4(mark, vid, iid, base);
    out.size = 1.0;
    return out;
}

fragment uint4 icb_merged_fragment(MergedOut in [[stage_in]]) {
    return in.ids;
}
