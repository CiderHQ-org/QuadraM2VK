// Owned fixture: a kernel whose only effect is the patch draws it
// encodes. Thread `tid` encodes command `tid`'s draw of one patch of one control point (patch 0,
// one instance, no patch index buffer), its triangle tessellation factors those at `factors`
// with no instance stride, scaled by `scales[tid]`. The pipeline and the vertex buffers are the
// executing encoder's, which the commands inherit. Fast math is off: a case's scale may be a NaN.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_render_patch_scale(
    device const Encoder &encoder [[buffer(0)]],
    const device MTLTriangleTessellationFactorsHalf *factors [[buffer(1)]],
    constant float *scales [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    render_command command(encoder.commands, tid);
    command.draw_patches(1, 0, 1, (const device uint *)nullptr, 1, 0, factors, 0, scales[tid]);
}
