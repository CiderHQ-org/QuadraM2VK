// A device cursor that is `null` on one edge of a merge and a real buffer offset on the other,
// over a buffer the emitter models RAW.
//
// The `uchar` view of `src` is what puts the buffer on the raw model, and then a non-zero GEP of
// it has no SPIR-V pointer value: the descriptor root and byte offset live only in the emitter's
// `raw_offsets`, and the merged cursor is re-materialized as a word index. The index phi refused
// any arm that was not a local, so the `null` arm collapsed the whole phi -- and every pointer
// downstream of it -- to a `Private` zero placeholder, and the guarded load answered 0.0 instead
// of the buffer.
//
// The null edge is never dereferenced: the shader's own `p != nullptr` check guards the load, so
// the index that edge contributes is unobservable and the answer only pins the LIVE edge.
//
// `bytes[3]` is the high byte of `src[0].x == 1.0f`, so it is 0x3F == 63 exactly -- a byte read
// INSIDE element 0, which no `float4`-typed access chain can express. `ctl[0]` is 1 so the live
// edge is taken and `ctl[1]` is 2, so the cursor is `src + 2` and `(*p).x` is 3.0. The answer is
// `3 + 63 + gid`: a cursor lost to the placeholder answers `63 + gid`, a cursor collapsed to the
// buffer root answers `64 + gid`, and a wrong vector component answers a large negative, since
// `src[k]` is `(k + 1, -100 - k, -200 - k, -300 - k)`. Every value is a small integer exact in
// binary32 and each thread writes only its own slot.
#include <metal_stdlib>
using namespace metal;

kernel void nullable_branch_cursor(device float4 *src [[buffer(0)]],
                                   const device uint *ctl [[buffer(1)]],
                                   device float *out [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    device const uchar *bytes = (device const uchar *)src;
    float base = float(bytes[3]);
    device float4 *p = nullptr;
    if (ctl[0] != 0u) {
        p = src + ctl[1];
    }
    float acc = 0.0f;
    if (p != nullptr) {
        acc = (*p).x;
    }
    out[gid] = acc + base + float(gid);
}
