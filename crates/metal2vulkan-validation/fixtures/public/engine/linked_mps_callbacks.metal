// Owned MSL: a kernel that calls, in the calling convention of the
// MPS NDArray kernels' linked callbacks, the visible functions the public linked modules under
// ../linked/ define. It builds the table of its four device buffers, reads its secondary and
// tertiary sources through `prefixSecondary_f` and `prefixTertiary_f` (element `tid` of buffers 1
// and 2: the source coordinate is `tid`'s base-4 digits), and stores four values through
// `postfixPrimary_4xf`, which writes them at element `4 tid` of the table's last buffer; the fourth
// value adds the bit count of `tid`, a 64-bit operation before the first local, so a declaration
// of a 64-bit type precedes the lifetime markers' (the decoded printout is then the same under
// every LLVM major). A case links both modules by their `.ll` hashes. Loop-free.
#include <metal_stdlib>
using namespace metal;

/// What the kernels pass a linked callback first: their table of device buffers and its length.
struct mps_buffer_table {
    device float * thread *buffers;
    ulong count;
};

[[visible]] void postfixPrimary_4xf(thread const mps_buffer_table &table,
                                    constant uchar &parameters, uint4 coordinate, uint count,
                                    float4 value);
[[visible]] float prefixSecondary_f(thread const mps_buffer_table &table,
                                    constant uchar &parameters, uint4 coordinate, uint count);
[[visible]] float prefixTertiary_f(thread const mps_buffer_table &table,
                                   constant uchar &parameters, uint4 coordinate, uint count);

kernel void linked_mps_callbacks(device float *primary [[buffer(0)]],
                                 device float *secondary [[buffer(1)]],
                                 device float *tertiary [[buffer(2)]],
                                 device float *destination [[buffer(3)]],
                                 constant uchar &parameters [[buffer(4)]],
                                 uint tid [[thread_position_in_grid]]) {
    const uint ones = uint(popcount(ulong(tid) << 33));
    device float *buffers[4] = {primary, secondary, tertiary, destination};
    const mps_buffer_table table = {buffers, 4};
    const uint4 source = uint4(tid & 3, (tid >> 2) & 3, (tid >> 4) & 3, tid >> 6);
    const float a = primary[tid];
    const float b = prefixSecondary_f(table, parameters, source, 4);
    const float c = prefixTertiary_f(table, parameters, source, 4);
    postfixPrimary_4xf(table, parameters, uint4(4 * tid, 0, 0, 0), 4, float4(a, b, c, a + b + c + float(ones)));
}
