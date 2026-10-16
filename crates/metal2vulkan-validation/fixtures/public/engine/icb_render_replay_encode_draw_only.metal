// Owned fixture: a kernel whose only effect is the draws it encodes.
// Thread `tid` encodes command `tid`'s draw of 4 points from vertex 4 * tid, one instance; the
// pipeline and the vertex buffers are the host's, or the executing encoder's when the commands
// inherit them. `params` and `out` are bound here unread, so a case owns the buffers the host
// binds to the commands or to the executing encoder. Only executing the commands observes them.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_render_replay_encode_draw_only(
    device const Encoder &encoder [[buffer(0)]],
    constant uint *params [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    render_command command(encoder.commands, tid);
    command.draw_primitives(primitive_type::point, 4 * tid, 4, 1, 0);
}
