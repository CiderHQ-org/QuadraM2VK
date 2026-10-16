// Owned fixture: load the last byte of a 6-byte `constant` buffer. `in` is a `constant` parameter
// whose air.arg_type_size (6) bounds its Window, so the translator admits a UBO view, and the
// std140 uvec4[] array is one of the UBO views it may select. This is the only input shape in which
// the padded allocation's uvec4 case is admissible (the SSBO uvec4[] view serves only
// 16-byte-aligned 16-byte leaves). The fixture alone does not force the uvec4 view: `u32[]` is
// equally valid wherever uniformBufferStandardLayout holds, which is every Vulkan 1.2 profile. Bind
// `in` to a 6-byte allocation. The UBO view of the padded descriptor range must still reach byte 5
// without leaving the descriptor: out[0] = in.b[5]. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Six {
    uchar b[6];
};

kernel void buffer_end_byte_load_constant(constant Six &in [[buffer(0)]],
                                          device uint *out [[buffer(1)]]) {
    out[0] = uint(in.b[5]);
}
