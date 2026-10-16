// Owned fixture: a kernel whose only effect is the patch draws it encodes,
// each of patches of one control point. Thread 0 encodes command 0's draw of 2 patches from
// position 1, each patch's index the `uint` at its position from the second of the device patch
// indices, twice from instance 1, their factors from the second triangle factor structure of the
// device factors on, scaled by `scales[0]`; thread 1 command 1's draw of one patch, its index the
// first constant patch index, once from instance 3, the constant factors, scaled by `scales[1]`;
// every other thread `t` command `t`'s draw of patch 0 with no patch index buffer, once from
// instance `t + 2`, the device factors, scaled by `scales[t]`. No factor instance stride. The
// pipeline and the vertex buffers are the executing encoder's, which the commands inherit;
// `marks` is bound here unread, so a case owns the buffer the encoder binds. Fast math is off: a
// case's scale may be a NaN. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_render_patch_encode(
    device const Encoder &encoder [[buffer(0)]],
    const device uint *patch_indices [[buffer(1)]],
    const device MTLTriangleTessellationFactorsHalf *factors [[buffer(2)]],
    constant float *scales [[buffer(3)]],
    constant uint *constant_indices [[buffer(4)]],
    constant MTLTriangleTessellationFactorsHalf *constant_factors [[buffer(5)]],
    device uint *marks [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    render_command command(encoder.commands, tid);
    if (tid == 0) {
        command.draw_patches(1, 1, 2, patch_indices + 1, 2, 1, factors + 1, 0, scales[0]);
    } else if (tid == 1) {
        command.draw_patches(1, 0, 1, constant_indices, 1, 3, constant_factors, 0, scales[1]);
    } else {
        command.draw_patches(1, 0, 1, (const device uint *)nullptr, 1, tid + 2, factors, 0,
                             scales[tid]);
    }
}
