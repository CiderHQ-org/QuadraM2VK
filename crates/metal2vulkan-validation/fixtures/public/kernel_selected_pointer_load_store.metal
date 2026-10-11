#include <metal_stdlib>
using namespace metal;

// Semantic twin of the hand-written AIR beside it. Every pointer here is CHOSEN at runtime: one is
// loaded through, one is stored through, and one is selected as a BASE and then indexed. Logical
// SPIR-V cannot merge pointers into two different descriptors into one value, so the translator has
// to replay each of those in the value domain instead -- which is the shape under test.
kernel void selected_pointer_load_store(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    bool lo = gid < 4;
    device const float* p = lo ? &a[gid] : &b[gid];
    float v = *p;
    device float* op = lo ? &out[gid] : &out[gid + 8];
    *op = v;
    device const float* base = lo ? b : a;
    out[gid + 16] = base[gid];
}
