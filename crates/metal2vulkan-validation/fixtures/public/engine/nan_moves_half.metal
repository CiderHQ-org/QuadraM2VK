// Owned fixture (NaN model): `nan_moves` on `half`, whether Metal's data movement
// keeps a NaN `half`'s bits. Lane `i` loads halves a and b (`hin[2i]`, `hin[2i + 1]`) and a
// selector, and stores as a half4 to `out[i]`: a (a half load and store), select(b, a, sel), its
// neighbour's a through threadgroup memory, and the selected element of half4(b, a, b, a).
// Compiled without fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_moves_half(device const half *hin [[buffer(0)]],
                           device const uint *sel [[buffer(1)]],
                           device half4 *out [[buffer(2)]],
                           uint tid [[thread_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]])
{
    threadgroup half tg[64];
    half a = hin[2 * tid], b = hin[2 * tid + 1];
    bool s = sel[tid] != 0;
    tg[lid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half t = tg[lid ^ 1u];
    half4 v = half4(b, a, b, a);
    out[tid] = half4(a, select(b, a, s), t, v[sel[tid] & 1u]);
}
