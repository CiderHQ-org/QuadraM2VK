// Owned fixture: a kernel whose only effect is the indexed draws it
// encodes. Thread 0 encodes command 0's draw of 5 points from 16-bit device indices, from the
// second, twice from instance 1, each index minus 2; thread 1 encodes command 1's draw of 3 points
// from 32-bit constant indices, once, each index plus 4. The pipeline is the executing encoder's,
// which the commands inherit, and the vertex buffers the host's. `params` and `out` are bound here
// unread, so a case owns the buffers the host binds to the commands. Only executing the commands
// observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_render_indexed_encode(
    device const Encoder &encoder [[buffer(0)]],
    const device ushort *short_indices [[buffer(1)]],
    constant uint *wide_indices [[buffer(2)]],
    constant uint *params [[buffer(3)]],
    device uint *out [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    render_command command(encoder.commands, tid);
    if (tid == 0) {
        command.draw_indexed_primitives(primitive_type::point, 5, short_indices + 1, 2, -2, 1);
    } else {
        command.draw_indexed_primitives(primitive_type::point, 3, wide_indices, 1, 4, 0);
    }
}
