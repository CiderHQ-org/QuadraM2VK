// Owned fixture: `discard_fragment` with demote semantics (MSL 2.3 and
// later; the compiler's default standard). Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Surface {
    float4 position [[position]];
    float4 color;
};

// A discard before the color output: demote where the profile has it, else predication (the
// entry's return kills a discarded invocation).
fragment float4 discard_demote(Surface in [[stage_in]]) {
    if (in.color.a < 0.5) {
        discard_fragment();
    }
    return in.color;
}

// A device store after the discard (fragmentStoresAndAtomics): a demoted invocation's store has
// no effect; predication would need the store guarded.
fragment float4 discard_then_store(Surface in [[stage_in]], device uint *out [[buffer(0)]]) {
    if (in.color.a < 0.5) {
        discard_fragment();
    }
    out[0] = 7u;
    return in.color;
}
