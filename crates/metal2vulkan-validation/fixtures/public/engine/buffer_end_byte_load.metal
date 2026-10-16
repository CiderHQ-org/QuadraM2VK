// Owned fixture: load the last byte of a 6-byte buffer. Bind `in` to a 6-byte allocation. `in` is a
// `device` pointer, so the translator selects an SSBO view: the u32[] word view (u8[] where 8-bit
// storage is present) of the padded descriptor range must still reach byte 5 without leaving the
// descriptor: out[0] = in[5]. The UBO (u32[]/uvec4) case is buffer_end_byte_load_constant.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void buffer_end_byte_load(device const uchar *in [[buffer(0)]],
                                 device uint *out [[buffer(1)]]) {
    out[0] = uint(in[5]);
}
