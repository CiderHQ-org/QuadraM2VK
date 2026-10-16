// Owned fixture: the compute template of commands whose stage-in region a
// kernel encodes. Each thread fetches its `[[stage_in]]` attribute (a case's descriptor: one uint
// stepping by thread position in the grid on x, so the element at the region's origin plus the
// thread's position) and reports it with the region's origin and size, at the slot the origin
// and its position name:
//     out[origin.x + t] = (v, origin.x | origin.y << 8 | origin.z << 16,
//                          size.x | size.y << 8 | size.z << 16, t)
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Sample {
    uint v [[attribute(0)]];
};

kernel void icb_stage_in_template(
    Sample in [[stage_in]],
    device uint4 *out [[buffer(0)]],
    uint t [[thread_position_in_grid]],
    uint3 origin [[grid_origin]],
    uint3 size [[grid_size]])
{
    out[origin.x + t] = uint4(in.v, origin.x | (origin.y << 8) | (origin.z << 16),
                              size.x | (size.y << 8) | (size.z << 16), t);
}
