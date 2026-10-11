#include <metal_stdlib>
using namespace metal;

// Two reinterpreting views of one `device uint *`: a byte cursor at offset 0, and a HALFWORD
// cursor at offset 4 taken through the byte cursor. `shorts[g]` is the halfword at byte
// `4 + 2*g` -- the view's own element stride, not the four bytes of the buffer it is carved out
// of. The two reads land in disjoint bit fields of the result, so a byte read that stayed correct
// still proves the store landed while the halfword moved.
kernel void kernel_byte_and_halfword_views(device uint *a [[buffer(0)]],
                                           device uint *o [[buffer(1)]],
                                           uint g [[thread_position_in_grid]]) {
    device uchar *bytes = (device uchar *)a;
    device ushort *shorts = (device ushort *)(bytes + 4);
    o[g] = uint(bytes[g]) | (uint(shorts[g]) << 16);
}
