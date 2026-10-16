// Owned MSL: linked source callbacks in the calling convention the
// MPS NDArray kernels call `prefixSecondary_f` and `prefixTertiary_f` with. A case links them (by
// this module's `.ll` hash) where the retained modules of those names return a constant.
#include <metal_stdlib>
using namespace metal;

/// What the kernels pass a linked callback first: their table of device buffers (the first field
/// of the kernel's `_MPSNDArrays`) and the table's length.
struct mps_buffer_table {
    device float * thread *buffers;
    ulong count;
};

/// Element `x + 4 (y + 4 (z + 4 w))` of table buffer `index`: a dense 4 x 4 x 4 layout of the
/// coordinate's lanes, the last lane unbounded.
static float element(thread const mps_buffer_table &table, uint index, uint4 coordinate) {
    return table.buffers[index][coordinate.x + 4 * (coordinate.y + 4 * (coordinate.z + 4 * coordinate.w))];
}

/// The secondary source: table buffer 1 at the coordinate.
[[visible]] float prefixSecondary_f(thread const mps_buffer_table &table,
                                    constant uchar &parameters, uint4 coordinate, uint count) {
    (void)parameters;
    (void)count;
    return element(table, 1, coordinate);
}

/// The tertiary source: table buffer 2 at the coordinate.
[[visible]] float prefixTertiary_f(thread const mps_buffer_table &table,
                                   constant uchar &parameters, uint4 coordinate, uint count) {
    (void)parameters;
    (void)count;
    return element(table, 2, coordinate);
}
