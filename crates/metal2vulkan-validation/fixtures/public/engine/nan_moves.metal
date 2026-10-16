// Owned fixture (NaN model): whether Metal's data movement keeps a NaN's bits. Lane
// `i` loads floats a and b (`fin[2i]`, `fin[2i + 1]`) and a selector, and stores as a float4 to
// `out[i]`: a (a float load and store), select(b, a, sel) (an if/else assignment folds to the same
// `select`), its neighbour's a through threadgroup memory, and the selected element of
// float4(b, a, b, a). `nan_moves_half` is the same program on `half`. Compiled without fast math.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_moves(device const float *fin [[buffer(0)]],
                      device const uint *sel [[buffer(1)]],
                      device float4 *out [[buffer(2)]],
                      uint tid [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float tg[64];
    float a = fin[2 * tid], b = fin[2 * tid + 1];
    bool s = sel[tid] != 0;
    tg[lid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = tg[lid ^ 1u];
    float4 v = float4(b, a, b, a);
    out[tid] = float4(a, select(b, a, s), t, v[sel[tid] & 1u]);
}
