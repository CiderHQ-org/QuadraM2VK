// Owned fixture: 16-byte buffer leaves aligned to 16, moved as `uvec4`s.
// The `uint4` loads of `w` and `src` and the stores to `out` lie at bytes 32 and up of their
// buffers, so the binder's descriptor ranges start 32 bytes in where the device's offset alignment
// is 16 and those leaves sit at a nonzero residual; the struct assignment is a
// 64-byte `align 16` copy from a `constant` buffer to `out[4..8]`. One thread writes
// out[2] = (src[2] & w.mask) + w.bias, out[3] = src[3] ^ w.bias, and out[4 + i] = rows.r[i].
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Weights {
    uint4 lead[2];
    uint4 mask;
    uint4 bias;
};

struct Rows {
    uint4 r[4];
};

kernel void vv1_sixteen_byte_units(constant Weights &w [[buffer(0)]],
                                   constant Rows &rows [[buffer(1)]],
                                   device const uint4 *src [[buffer(2)]],
                                   device uint4 *out [[buffer(3)]]) {
    *(device Rows *)(out + 4) = rows;
    out[2] = (src[2] & w.mask) + w.bias;
    out[3] = src[3] ^ w.bias;
}
