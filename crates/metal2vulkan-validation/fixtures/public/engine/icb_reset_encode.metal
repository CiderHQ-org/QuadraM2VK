// Owned fixture: a kernel whose only effect is resetting commands the host
// encoded. Thread 0 resets compute command 1 of the first command buffer; every other thread
// resets render command 0 of the second. Each command's other state is the host's; a reset one
// runs nothing. `params` and `out` are bound here unread, so a case owns the buffers the host
// binds to the commands. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer compute_commands;
    command_buffer render_commands;
};

kernel void icb_reset_encode(
    device const Encoder &encoder [[buffer(0)]],
    constant uint *params [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        compute_command command(encoder.compute_commands, 1);
        command.reset();
    } else {
        render_command command(encoder.render_commands, 0);
        command.reset();
    }
}
