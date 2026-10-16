// Owned fixture: a kernel whose only effect is the indexed patch
// draws it encodes. Thread `tid` encodes command `tid`'s indexed patch draw of `draws[tid]`'s
// control-point count, patch start and count, instance count and base instance, with no factor
// instance stride and `draws[tid]`'s factor scale: thread 0's patch indices (from `uint`
// `patch_offset`), control-point indices (from byte `point_offset`) and factors (from structure
// `factor_offset`) in device memory, thread 1's in constant memory; thread 2 with no patch index
// buffer, its control-point indices and factors in device memory; every other thread with neither
// a patch index nor a control-point index buffer, its factors in device memory. The pipeline and
// the vertex buffers are the executing encoder's, which the commands inherit; `marks` is bound
// here unread, so a case owns the buffer the encoder binds. Fast math is off: a draw's scale may
// be a NaN. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

struct Draw {
    uint control_points;
    uint patch_start;
    uint patch_count;
    uint instance_count;
    uint base_instance;
    uint patch_offset;
    uint point_offset;
    uint factor_offset;
    float scale;
};

kernel void icb_render_ipatch_encode(
    device const Encoder &encoder [[buffer(0)]],
    const device uint *patch_indices [[buffer(1)]],
    const device uchar *point_indices [[buffer(2)]],
    const device MTLTriangleTessellationFactorsHalf *factors [[buffer(3)]],
    constant Draw *draws [[buffer(4)]],
    constant uint *constant_patch_indices [[buffer(5)]],
    constant uchar *constant_point_indices [[buffer(6)]],
    constant MTLTriangleTessellationFactorsHalf *constant_factors [[buffer(7)]],
    device uint *marks [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    constant Draw &d = draws[tid];
    render_command command(encoder.commands, tid);
    if (tid == 0) {
        command.draw_indexed_patches(d.control_points, d.patch_start, d.patch_count,
                                     patch_indices + d.patch_offset,
                                     point_indices + d.point_offset, d.instance_count,
                                     d.base_instance, factors + d.factor_offset, 0, d.scale);
    } else if (tid == 1) {
        command.draw_indexed_patches(d.control_points, d.patch_start, d.patch_count,
                                     constant_patch_indices + d.patch_offset,
                                     constant_point_indices + d.point_offset, d.instance_count,
                                     d.base_instance, constant_factors + d.factor_offset, 0,
                                     d.scale);
    } else if (tid == 2) {
        command.draw_indexed_patches(d.control_points, d.patch_start, d.patch_count,
                                     (const device uint *)nullptr,
                                     point_indices + d.point_offset, d.instance_count,
                                     d.base_instance, factors + d.factor_offset, 0, d.scale);
    } else {
        command.draw_indexed_patches(d.control_points, d.patch_start, d.patch_count,
                                     (const device uint *)nullptr, (const device void *)nullptr,
                                     d.instance_count, d.base_instance,
                                     factors + d.factor_offset, 0, d.scale);
    }
}
