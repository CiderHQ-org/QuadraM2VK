// Each thread encodes one render command -- a vertex buffer, a fragment buffer, and a draw -- and
// writes an output the command does not depend on. Both executors run the encoding under the
// authored descriptor policy; the encoded commands are not observed (no Metal readback; replay is
// owed), so a case observes the output buffer only.
#include <metal_stdlib>
using namespace metal;

struct Commands {
    command_buffer commands;
};

kernel void kernel_icb_encode_render_commands(
    device const Commands &encoder [[buffer(0)]],
    device float4 *vertices [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    render_command command(encoder.commands, tid);
    command.set_vertex_buffer(vertices, 0);
    command.set_fragment_buffer(vertices, 1);
    command.draw_primitives(primitive_type::triangle, tid * 3, 3, 1, 0);
    out[tid] = tid * 5 + 2;
}
