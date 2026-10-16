// Owned fixture: load the last half (bytes 8..9) of a 10-byte buffer.
// Bind `in` to a 10-byte allocation: out[0] = in[4]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void buffer_end_half_load(device const ushort *in [[buffer(0)]],
                                 device uint *out [[buffer(1)]]) {
    out[0] = uint(in[4]);
}
