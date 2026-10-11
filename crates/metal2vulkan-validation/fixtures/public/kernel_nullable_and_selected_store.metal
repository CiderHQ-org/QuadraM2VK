#include <metal_stdlib>
using namespace metal;

// Semantic twin of the hand-written AIR beside it. Two shapes the translator cannot express with a
// pointer value: a pointer that is NULL on half the lanes and is then tested against null, and a
// STORE through a base pointer chosen between two different descriptors and then indexed. Metal only
// has to agree on the value.
kernel void nullable_and_selected_store(
    device float* c [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    bool lo = gid < 4;
    device float* maybe = lo ? &c[gid] : nullptr;
    out[gid] = (maybe == nullptr) ? 1.0f : 0.0f;

    device float* wbase = lo ? c : out;
    uint k = gid + 8;
    wbase[k] = float(k);
    out[gid + 16] = c[k];
}
