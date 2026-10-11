#include <metal_stdlib>
using namespace metal;

// A helper the frontend must not inline, so the AIR keeps the call and the translator's
// pointer-select consumer inliner is the one that removes it.
__attribute__((noinline))
static float read_at(device const float *p, uint i) { return p[i]; }

__attribute__((noinline))
static uint read_word(device const uchar *p, uint byte_offset) {
    return *(device const uint *)(p + byte_offset);
}

kernel void ptrsel(device const float *a [[buffer(0)]],
                   device const float *b [[buffer(1)]],
                   device const uchar *raw [[buffer(3)]],
                   device float *out [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    bool take_a = (gid & 1u) != 0u;

    // 1. A ternary pointer select between two DISTINCT device buffers, consumed by a
    //    non-inlined helper -- the shape inline_pointer_select_consumers fires on.
    device const float *p = take_a ? a : b;
    out[gid] = read_at(p, gid);

    // 2. The same merged pointer indexed at a runtime offset, so the GEP is taken off the
    //    select rather than off either arm.
    out[gid + 8] = p[(gid + 3u) & 7u];

    // 3. A pointer PHI: the two arms do different work before joining, so the frontend
    //    cannot narrow this to a select.
    device const float *q;
    if (take_a) {
        q = a + (gid & 3u);
    } else {
        q = b + ((gid >> 1) & 3u);
    }
    out[gid + 16] = q[0];

    // 4. The merged pointer read a second time, at a different index, so the GEP off the
    //    phi is taken twice from one merge.
    out[gid + 24] = q[1] + read_at(q, 2u);

    // 5. A raw byte-addressed select: the merged pointer is a byte pointer, so the word
    //    index comes off the select.
    device const uchar *r = take_a ? raw : (raw + 16);
    out[gid + 32] = as_type<float>(read_word(r, (gid & 3u) * 4u));
}
