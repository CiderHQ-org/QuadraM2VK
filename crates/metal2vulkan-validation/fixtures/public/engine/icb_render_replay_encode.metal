// Owned fixture: a rasterization-disabled vertex function whose only
// effect is the render commands it encodes, the shape of a GPU validation layer's patch pass.
// Vertex `vid` below the execution range's length encodes command `range.location + vid`: the
// constant `params` window and the device `out` window at 256 * vid bytes as vertex buffers 0
// and 1, and `params` as fragment buffer 0. Each command's draw is the host's, and its pipeline
// the executing encoder's. Only executing the commands observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

vertex void icb_render_replay_encode(
    device const Encoder &encoder [[buffer(0)]],
    constant uint *params [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant MTLIndirectCommandBufferExecutionRange &range [[buffer(3)]],
    uint vid [[vertex_id]])
{
    if (vid >= range.length) {
        return;
    }
    render_command command(encoder.commands, range.location + vid);
    command.set_vertex_buffer(params + vid * 64, 0);
    command.set_vertex_buffer(out + vid * 64, 1);
    command.set_fragment_buffer(params, 0);
}
