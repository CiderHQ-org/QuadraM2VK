#include <metal_stdlib>
using namespace metal;

// Integer division and remainder in the two shapes the denominator guard distinguishes:
// a RUNTIME denominator (the guard must stay) and the emitter's own constant word-index
// divide behind a raw byte offset (the guard folds away). Every divisor here is non-zero.
kernel void divshapes(device const uint *in [[buffer(0)]],
                      device const uchar *raw [[buffer(2)]],
                      device uint *out [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint  ua = in[gid];
    uint  ub = in[gid + 8];          // runtime, never zero
    int   sa = as_type<int>(ua);
    int   sb = as_type<int>(ub);

    out[gid +  0] = ua / ub;
    out[gid +  8] = ua % ub;
    out[gid + 16] = as_type<uint>(sa / sb);
    out[gid + 24] = as_type<uint>(sa % sb);

    uint2 va = uint2(ua, ua >> 3);
    uint2 vb = uint2(ub, ub | 1u);
    uint2 vq = va / vb;
    out[gid + 32] = vq.x;
    out[gid + 40] = vq.y;

    ulong la = (ulong(ua) << 20) | ulong(ub);
    ulong lb = ulong(ub) + 1;
    out[gid + 48] = uint(la / lb);
    out[gid + 56] = uint(la % lb);

    // Raw byte offset: the translator turns this into a word index with its own `/ 4`.
    uint byte_off = (ua & 7u) * 4u;
    device const uint *w = (device const uint *)(raw + byte_off);
    out[gid + 64] = *w;
}
