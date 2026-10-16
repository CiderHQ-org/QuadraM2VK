// Owned MSL: a linked destination callback in the calling convention
// the MPS NDArray kernels call `postfixPrimary_4xf` with. A case links it (by this module's `.ll`
// hash) where the retained modules of that name write nothing.
#include <metal_stdlib>
using namespace metal;

/// What the kernels pass a linked callback first: their table of device buffers (the first field
/// of the kernel's `_MPSNDArrays`) and the table's length.
struct mps_buffer_table {
    device float * thread *buffers;
    ulong count;
};

/// Stores `value` at elements `coordinate.x` to `coordinate.x + 3` of the table's last buffer,
/// `count` being the table's length.
[[visible]] void postfixPrimary_4xf(thread const mps_buffer_table &table,
                                    constant uchar &parameters, uint4 coordinate, uint count,
                                    float4 value) {
    (void)parameters;
    device float *destination = table.buffers[count - 1];
    destination[coordinate.x] = value.x;
    destination[coordinate.x + 1] = value.y;
    destination[coordinate.x + 2] = value.z;
    destination[coordinate.x + 3] = value.w;
}
